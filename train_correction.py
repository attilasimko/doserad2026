import os as _os
# pymedphys._interp is numba-jitted with cache=True, and numba writes its JIT
# cache (.nbi/.nbc) INSIDE the installed package directory. On a cluster the venv
# usually is not writable from the compute node, and the gamma call then dies with
# FileNotFoundError before it computes anything. Redirect the cache somewhere
# writable BEFORE pymedphys (hence numba) is imported.
_os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    _os.path.join(_os.environ.get("TMPDIR") or "/tmp",
                  f"numba_cache_{_os.environ.get('USER', 'user')}"))
_os.makedirs(_os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

from comet_ml import Experiment, OfflineExperiment
import json
import pydosert as PDRT
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from configs import machine_config, DEFAULT_KERNEL_SIZE, MULTISLAB_DEFAULTS, default_data_path
import logging
from natsort import natsorted
import os
import sys
import math
import heapq
import torch.nn as nn
import torch.optim as optim
import argparse
import re

try:
    import pymedphys
    _HAS_PYMEDPHYS = True
except ImportError:
    pymedphys = None
    _HAS_PYMEDPHYS = False
from loaders import (
    BeamDataset,
    crop_and_pad_to_original,
    SameBeamBatchSampler,
    same_beam_collate,
)
from plotting import plot_dose_comparison
import time
from torch.utils.data import DataLoader
from engines import warm_start_from, CorrectedDoseEngine, FluenceCorrectionModel, DoseCorrectionModel

logging.getLogger().setLevel(logging.ERROR)
os.environ["TQDM_DISABLE"] = "1"

# Modality-specific warm-start checkpoints. Every training run loads the
# corresponding file into the dose_correction_model before the optimizer
# is built, so each sweep job continues from these baselines instead of
# starting from random init. Override or skip with --no_warm_start.
# Paths match the ckpt save format: out/<comet_run_name>_best.pt.
# Both arms warm-start from the SAME checkpoint: after the MR is converted to
# a synthetic CT the corrector sees identical inputs in both cases, so there is
# no such thing as an "MR corrector" any more. (The old mri_photon.pt was
# trained with the raw MR as the corrector's support channel — that input no
# longer exists, so it is not a valid starting point.)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None,
    help="Dataset root (the dir containing photon/ and proton/). "
         "Defaults to $DOSERAD_DATA.")
parser.add_argument("--l1_alpha", type=float, default=0.0)
parser.add_argument("--lrate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=101)
parser.add_argument(
    "--modality", type=str, default="ct",
    help="Modalities to TRAIN on. Comma-separated, e.g. 'ct,mr'. The corrector "
         "is deployed on BOTH real CT and MR-derived synthetic CT, so training "
         "on ct alone leaves the MR arm a pure generalisation gap. BeamDataset "
         "indexes each (patient, beam, cp, modality) separately and "
         "SameBeamBatchSampler keys on modality too, so a batch never mixes "
         "arms. REQUIRES precomputed sct.mha for the TRAINING cohort "
         "(precompute_sct.py --cohort training); without it every sample load "
         "would run the sCT network live.")
parser.add_argument("--treatment", type=str, default="photon")
parser.add_argument(
    "--anatomies",
    type=str,
    default="both",
    choices=["thorax", "abdomen", "both"],
    help="Debug filter only. Production always trains on 'both': thorax and "
         "abdomen share one corrector (the calibrated engine is anatomy-agnostic).",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default=None,
    help="Explicit name for this run: sets the Comet experiment name AND the "
         "checkpoint basename (out/<run_tag>_best.pt) instead of a random Comet "
         "name, so the four per-anatomy models land in predictable files "
         "(e.g. --run_tag ct_thorax -> out/ct_thorax_best.pt).",
)
parser.add_argument(
    "--patient",
    type=str,
    default=None,
    help="Optional single patient id (e.g. '1ABB020') to overfit/debug on. "
         "Takes precedence over --anatomies for the patient list but anatomy filter still applies.",
)
parser.add_argument(
    "--v2_w_local", type=float, default=0.0,
    help="Weight on a LOCAL-relative L1 term added to --v2_loss: "
         "|pred-target| / max(target, 0.1*peak), averaged over target >= "
         "0.1*peak. That is exactly the denominator and the scored set local "
         "gamma uses. The support and high10 terms are normalised by the "
         "per-sample PEAK, which makes them MAE-shaped: they weight a 3%% error "
         "at 12%% of Dmax the same in absolute terms as at 100%%, where gamma "
         "judges both against their own local dose. This is the opposite pole.")
parser.add_argument(
    "--stem_stride", default="1",
    help="Downsample the TRUNK input by this factor and resample its "
         "correction back up; the refine branch stays at full resolution. "
         "Accepts an int (isotropic) or a D:H:W triple, e.g. '2:1:1'. BEV is "
         "[B,C,D,H,W] with D the BEAM DEPTH -- the smooth axis (build-up plus "
         "an exponential) -- while H and W carry the ~5 mm lateral penumbra. "
         "Isotropic 2 discards 8x the voxels INCLUDING lateral and FAILED on "
         "accuracy for exactly that reason. '2:1:1' halves the trunk's voxels "
         "by pooling only the smooth axis.")
parser.add_argument(
    "--refine_hidden", type=int, default=16,
    help="Width of the full-resolution dilated refine branch. It is the more "
         "expensive half of the corrector, not the U-Net: measured 92.5 ms "
         "against the trunk's 77.9 ms at the default 16. Halving it to 8 is a "
         "bigger single lever than --stem_stride 2.")
parser.add_argument(
    "--tiny_fluence_ckpt",
    default="models/tiny_fluence_tile_mean_epoch1.pt",
    help="Frozen TinyFluenceCorrector checkpoint used as part of the fixed "
         "physics prior before training the dose corrector. ON BY DEFAULT: it "
         "is 39k frozen parameters that improve the baseline the corrector "
         "learns against, so leaving it off silently trains against a weaker "
         "engine. Pass 'none' to disable.",
)
parser.add_argument(
    "--lateral_heterogeneity_ckpt",
    default="models/tiny_lateral_heterogeneity_tile_mean_fluence_e1.pt",
    help="Frozen tiny lateral heterogeneity checkpoint applied after TERMA. "
         "This is an engine prior, not part of the trainable corrector. ON BY "
         "DEFAULT: 48 frozen parameters, measured -6.3%% beam MAE and -6.9%% IDD "
         "on top of the fluence prior, and disproportionately on thorax. "
         "REQUIRES --terma (it was trained to sit after it); pass 'none' to "
         "disable.",
)
parser.add_argument("--band_weights", type=str, default=None,
                    help="Comma-separated weights for the l1_banded dose bands "
                         "(high>=80%%, mid 30-80%%, low 10-30%%, periphery 2-10%%). "
                         "Default 1,1,1,0.5.")
parser.add_argument("--val_modalities", type=str, default="ct,mr",
                    help="Modalities to VALIDATE on. Training always uses --modality "
                         "(real CT); validation covers both arms in one pass.")
parser.add_argument("--gamma_every_n_epochs", type=int, default=5,
                    help="Run the plan gamma every Nth epoch (and always on the last). "
                         "pymedphys.gamma is numpy on CPU, so ~350 s per plan x 12 plans "
                         "= ~70 min per validation cycle regardless of GPU — paying that "
                         "every epoch dominates the run. 1 = every epoch, 0 = never.")
parser.add_argument("--gamma_beam", type=int, default=0,
                    help="Beam index whose summed control points get a full-grid gamma. "
                         "One evaluation per (patient, modality) -> 12 per epoch. "
                         "-1 = every beam. Gamma is never computed per control point.")
parser.add_argument(
    "--num_logged_beams",
    type=int,
    default=8,
    help="Number of validation beams to log to Comet each epoch "
         "(probe beam + worst-MAE + stratified random).",
)
parser.add_argument(
    "--num_beams",
    type=int,
    default=3,
    help="Number of beams (0..n) per patient to iterate over (dataset has 3). "
         "Default 3 mirrors the actual challenge problem; --max_train_steps "
         "bounds training cost regardless.",
)
parser.add_argument(
    "--val_num_beams", type=int, default=None,
    help="Number of beams to VALIDATE on (0..n). Default = --num_beams. Set to 1 to validate "
         "on beam 0 only — validation is slow, and beam 0 is representative for tracking lvl1.",
)
parser.add_argument(
    "--num_cps",
    type=int,
    default=180,
    help="Number of control points per beam to iterate over.",
)
parser.add_argument(
    "--cp_stride",
    type=int,
    default=1,
    help="Stride over control points (e.g. 4 → every 4th CP, evenly spaced).",
)
parser.add_argument("--v2_alpha", type=float, default=0.05,
                    help="Additive-head bound as a fraction of the CP peak.")
parser.add_argument(
    "--v2_cond_mode", choices=("channels", "off"), default="channels",
    help="How the 3 conditioning scalars (log aperture area, mean rho_e, lung "
         "fraction) reach the corrector. USE 'channels': the scalars are "
         "broadcast as 3 extra input channels, which is what the best run "
         "does and what takes the stack from 10 to 13 channels. 'film' "
         "emitted a separate cond vector for the deferred v2 trunk's FiLM "
         "layers; the v1 trunk has none, so with --v2_features it is "
         "equivalent to 'off' and silently DROPS the conditioning. Kept as "
         "the default only so older command lines keep their exact meaning; "
         "a warning is printed.")
parser.add_argument(
    "--warm_start", type=str, default=None,
    help="Checkpoint to initialise the corrector from. Copies every tensor "
         "matching by NAME AND SHAPE and reports what was inherited vs left at "
         "init, so an architecture change degrades to a partial load rather "
         "than failing or silently loading garbage. Absent = random init.")
parser.add_argument(
    "--refine_mode", choices=("parallel", "sequential"), default="parallel",
    help="How the full-resolution refine branch combines with the trunk. "
         "'parallel' (shipped): the refiner reads the RAW input and its output "
         "is added to the trunk's correction -- neither branch sees the "
         "other's answer, so both infer the correction independently and their "
         "SUM has to be right. 'sequential': the refiner also receives the "
         "trunk's correction as a peak-normalised extra channel and predicts a "
         "residual on top, i.e. a cascade that can fix what the trunk got "
         "wrong. Costs 432 extra parameters.")
parser.add_argument(
    "--adam_eps", type=float, default=1e-8,
    help="Adam/AdamW epsilon. The optimiser runs in fp32 (bf16 autocast covers "
         "only the corrector forward), so this is not an underflow guard -- it "
         "matters only relative to sqrt(v). Measured on a real backward: grad "
         "rms 1.1e-07, per-tensor rms 4.4e-07 to 9.6e-05. At the default 1e-8 "
         "eps is ~2%% of the QUIETEST tensor's sqrt(v), so it damps exactly the "
         "parameters that are learning slowest. 1e-10 makes it negligible "
         "everywhere and lets those tensors take full Adam-normalised steps.")
parser.add_argument("--v2_w_high10", type=float, default=0.15)
parser.add_argument(
    "--v2_w_idd", type=float, default=0.0,
    help="Depth-profile L1 term. REOPENED 2026-08-26. It was closed because it "
         "degrades Level-1 MAE -- true, and it no longer settles the question: "
         "under RankThenMean, beam MAE and IDD are ONE SLOT EACH, and on the MR "
         "board IDD ranks 25 against beam MAE's 21. Giving up MAE rank for IDD "
         "rank is a net gain there. Default 0.0 = LossConfig default = off.")
parser.add_argument("--v2_w_deep", type=float, default=0.1)
parser.add_argument(
    "--v2_w_mse", type=float, default=0.0,
    help="Peak-normalised MSE term. Both published MR-linac dose networks use a "
         "squared error (Tsekas 2022 RMSE, Tseng 2023 MSE) where this loss is "
         "MAE-based. MSE weights a 4x error 16x, and gamma failures are the "
         "large errors. 0.0 = off = LossConfig default.")
# --- LOW-DOSE EMPHASIS. All default 0.0 = LossConfig default = OFF, so an
# unchanged command line trains exactly the shipped loss. See losses_v2 for the
# measurement that motivates them: ~90% of local-gamma failures sit in the
# 10-30% dose band, and stratified plan MAE weights that band as a full third
# of the score.
parser.add_argument(
    "--v2_w_strata", type=float, default=0.0,
    help="Equal-weight dose-band MAE (10-30 / 30-80 / >80%% of sample peak). "
         "Mirrors stratified plan MAE, which averages its three bands equally "
         "regardless of voxel count.")
parser.add_argument(
    "--v2_w_rel", type=float, default=0.0,
    help="Relative error |p-t|/t over scored voxels -- exactly what local gamma "
         "1%%/1mm thresholds. Denominator floored at 5%% of peak.")
parser.add_argument(
    "--v2_w_logw", type=float, default=0.0,
    help="Log-weighted MAE: weight 1 at peak growing as log(peak/t) toward low "
         "dose, capped at 6. At 2.0 the low-band gradient is 1.83x the "
         "high-band, against 1.00x unweighted.")
parser.add_argument(
    "--feature_set", choices=("v2", "v3", "v4"), default="v2",
    help="Which --v2_features stack to build. 'v2' is the 12-channel stack "
         "photon_corrector.pt was trained on and stays the default. 'v3' drops "
         "depth_index (crop-relative, so its meaning changes with the BEV box) "
         "and adds the projected aperture as a continuous peak-normalised "
         "fluence channel -- the engine already computes it for the physics, "
         "so it is free. Both sets emit 12 channels, so the CHANNEL COUNT "
         "CANNOT tell them apart: the name is written into model_config and "
         "checked at load.")
parser.add_argument(
    "--cond_region", choices=("dose", "aperture"), default="dose",
    help="Which voxels cond_mean_rho / cond_lung_frac average over. 'dose' is "
         "the original (dose > 10%% of peak) and is WRONG: air is rho_e ~ 0 and "
         "so counts as lung, and the pencil beam deposits above the cutoff "
         "outside the patient, so on abdomen the channels measure empty space "
         "-- 1ABB045 reads lung_frac 0.370 of which 0.354 is air. 'aperture' "
         "is the open field intersected with the body over the depth the dose "
         "covers, which restores the cohort separation the channels exist for "
         "(abdomen 0.005-0.026 vs thorax 0.280-0.474). The default is the old "
         "one because it is what every existing checkpoint was trained on and "
         "swapping it under a trained model changes two of its thirteen input "
         "channels by more than their whole between-cohort range.",
)
parser.add_argument(
    "--v1_material_embed_dim", type=int, default=0,
    help="Width of the Geant4 composition-family embedding (86 classes) fed to "
         "the v1 trunk alongside the scalar stack. 0 = off (the shipped "
         "behaviour). Carries what electron density cannot: ~half the fluence "
         "is above the 1.022 MeV pair threshold and pair production is "
         "Z-dependent. Expect it to help BONE and do nothing for LUNG, which is "
         "one family across -950..-90 HU. Needs --v2_features for the "
         "material id to exist.")
parser.add_argument(
    "--v1_deep_supervision", action="store_true",
    help="Zero-init aux heads on the coarse decoder levels of the v1 trunk, "
         "scored by the existing --v2_loss deep term at --v2_w_deep. Until now "
         "the v1 trunk emitted no aux predictions, so w_deep was dead code and "
         "every run that set it trained without it.")
parser.add_argument(
    "--seed", type=int, default=None,
    help="Seed torch / numpy / random and the DataLoader workers. NOTHING was "
         "seeded before this: weight init AND batch order were both random per "
         "run, so two runs of an identical config landed 16%% apart on Level-1 "
         "and 55%% apart on Level-2 (jobs 556359 vs 556407, epoch 1). Every "
         "single-run A/B in this repo carries that noise. Pass the same seed "
         "to both arms of a comparison. Note this fixes init and data order "
         "but not GPU kernel non-determinism -- see --deterministic.")
parser.add_argument(
    "--deterministic", action="store_true",
    help="Also force deterministic GPU kernels (cudnn.deterministic, "
         "use_deterministic_algorithms). Needed for bit-exact reruns on the "
         "same hardware; costs throughput, so it is opt-in on top of --seed.")
parser.add_argument(
    "--amp", action="store_true",
    help="Autocast the CORRECTOR forward (not the physics engine). No photon "
         "arm has ever trained on a 40 GB card and base_channels 16 OOM'd even "
         "80 GB, so memory -- not ideas -- has been the binding constraint. "
         "The proton path has run bf16 by default all along; this repo has "
         "been pure fp32.")
parser.add_argument(
    "--amp_dtype", choices=("bfloat16", "float16"), default="bfloat16",
    help="bfloat16 keeps fp32 exponent range, so it needs no GradScaler and "
         "cannot overflow on the small dose magnitudes here (~7e-5 per CP "
         "before the x1e5 gain). float16 is offered only for comparison and "
         "would need loss scaling that is NOT implemented.")
parser.add_argument(
    "--grad_clip", type=float, default=0.0,
    help="Clip global grad norm to this value; 0 disables. OFF by default so "
         "every earlier arm stays reproducible, but it should be on for "
         "anything combining an unbounded output head with the peak-normalised "
         "v2 loss -- that pairing destroyed job 556220 at epoch 4 (train loss "
         "0.0064 -> 0.0144, validation back to the untrained baseline). The "
         "proton path clips, and logs `clip=` every step.")
parser.add_argument(
    "--v1_norm", choices=("batch", "group", "group_masked", "instance", "none"),
    default="batch",
    help="Normalisation in the v1 trunk and refine branch. BatchNorm is the "
         "historical default and a poor fit: a batch is batched_cp_size "
         "control points from the SAME beam, 2 degrees apart, so the "
         "statistics come from correlated samples -- and training uses 3 beams "
         "while validation uses 1, so running stats are gathered on one "
         "distribution and applied to another. "
         "'group_masked' is GroupNorm with the statistics taken over the "
         "APERTURE only. Plain GroupNorm averages over the whole box, and "
         "under --bev_crop the box is the field plus a fixed margin, so the "
         "in-field fraction -- and therefore the mean and the variance -- "
         "moves with the field size at every control point. Masking makes "
         "the reference the dose being corrected rather than the box it was "
         "cut from. Needs the projected aperture, so it requires the "
         "pencil-beam path.")
parser.add_argument(
    "--v1_bounded_residual", action="store_true",
    help="Give v1 the v2 output algebra: additive head bounded to "
         "--v2_alpha times the per-sample peak, and relu on the summed dose. "
         "Removes the blow-up mode structurally rather than only clipping it.")
parser.add_argument(
    "--val_cp_stride", type=int, default=1,
    help="Subsample control points in VALIDATION. Default 1 = every CP, which "
         "is what plan reconstruction needs; --cp_stride only ever subsampled "
         "training. Raise it for a quick paired measurement on one machine, "
         "where the CP set is identical on both sides and only the variable "
         "under test differs.")
parser.add_argument(
    "--v2_loss", action="store_true",
    help="Use the v2 composite loss (peak-normalised support MAE + 0.15 x "
         "high10) instead of --loss. Worth 30%% on Level-1 and 34%% on "
         "Level-2 against plain L1 on the same 13-channel stack, and it is "
         "what the best run trains with. Independent of the feature stack, so "
         "objective and inputs can be ablated separately.")
parser.add_argument(
    "--v2_features", action="store_true",
    help="Feed the v2 BEV feature stack (features_v2.build_bev_features) to "
         "the corrector. Channel 0 stays the raw baseline dose, which v1's "
         "gain head multiplies and whose units the engine adds back, so "
         "substituting the peak-normalised channel 0 would break the "
         "correction algebra; the v2 channels are appended after it. With "
         "--v2_cond_mode channels that is 1 + 12 = 13 input channels, which "
         "is the best run.")
parser.add_argument(
    "--base_channels",
    type=int,
    default=4,
    help="Base channel count for the DoseCorrectionModel U-Net. "
         "Memory roughly scales linearly. 4 is the safe default for A40; "
         "8 is reasonable on G=1 only.",
)
parser.add_argument(
    "--unet_depth",
    type=int,
    default=4,
    help="Number of pooling levels in the corrector U-Net.",
)
parser.add_argument(
    "--compact_unet",
    action="store_true",
    help="Use the compact no-refiner U-Net preset: base width 4, depth 3, "
         "depth-only learned downsampling, 1.4x width growth, separable "
         "fine-level convolutions, activation checkpointing, and a 0.15 "
         "peak-bounded residual. The resolved fields are stored separately "
         "in model_config so inference does not depend on this shortcut.",
)
parser.add_argument(
    "--no_gain_head",
    action="store_true",
    help="Disable the multiplicative gain head (residual-only correction).",
)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="If set, cap each training epoch to this many iterations so "
         "validation runs more often. The train loader is shuffled, so "
         "each epoch still sees a different random subset.",
)
parser.add_argument(
    "--profile_steps",
    type=int,
    default=0,
    help="If >0, print a per-section timing breakdown (data, "
         "engine init, forward, backward, optim) every N training "
         "steps. Adds CUDA syncs around the timed sections so leave "
         "it OFF (0) for production runs.",
)
parser.add_argument(
    "--skip_validation",
    action="store_true",
    help="Skip the validation pass entirely. Useful when debugging "
         "training throughput — validation is the longest non-training "
         "section of an epoch.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="DataLoader workers. Set to 0 to load synchronously in the "
         "main process — disambiguates whether a hang lives in worker "
         "IPC / fork-after-CUDA vs the collate / dataset itself.",
)
parser.add_argument(
    "--loss",
    choices=("l1", "l2", "smooth_l1", "l1_grad", "l2_grad", "l1_masked", "huber_grad",
             "multiscale", "l1_banded", "l1_banded_grad"),
    default="l1",
    help="Training loss. 'l1' = mean absolute error. 'l2' = mean squared "
         "error (the textbook regression baseline; over-weights high-dose "
         "hotspots vs L1). 'smooth_l1' = Huber (gentler at small errors). "
         "'l1_grad' = L1 + grad_alpha * L1 on dose gradients (penalises "
         "penumbra / interface blur). 'l2_grad' = MSE + grad_alpha * "
         "gradient-L1 (apples-to-apples version of l1_grad with an L2 base "
         "term). 'l1_masked' = L1 restricted to voxels >=10%% of the "
         "target peak — directly mirrors the challenge's masked-MAE "
         "metric so capacity isn't spent on the scored-out periphery. "
         "'huber_grad' = smooth_l1 + grad_alpha * gradient-L1. "
         "'multiscale' = L1 at full + 1/2 + 1/4 resolution plus "
         "grad_alpha * gradient-L1; targets the local-gamma structure "
         "without the instability of a large single-scale gradient weight.",
)
parser.add_argument(
    "--attention",
    choices=("none", "se", "sa"),
    default="none",
    help="Corrector U-Net attention. 'none' = plain. 'se' = squeeze-"
         "excitation channel attention in every conv block (cheap). "
         "'sa' = SE blocks plus a self-attention layer at the "
         "bottleneck (more expressive, heavier).",
)
parser.add_argument(
    "--use_source_distance",
    action="store_true",
    help="Feed a per-voxel distance-to-source (PSF prior) as a 3rd "
         "corrector input channel. Computed geometrically in BEV: "
         "sqrt(depth^2 + lat_h^2 + lat_w^2) with the source at SAD "
         "along the beam direction, normalised by SAD so the value is "
         "~1 at iso. Gives the model a structured spatial prior of "
         "where the iso is, where the beam is going, and how dose "
         "falls off geometrically (1/r^2 lives one square away).",
)
parser.add_argument(
    "--pool_kernels", default=None,
    help="Per-level MaxPool3d kernels for the U-Net trunk, as D:H:W triples "
         "separated by commas, e.g. '2:1:1,2:1:1,2:2:2,2:2:2'. Default (unset) "
         "is the isotropic 2:2:2 at every level that every existing checkpoint "
         "uses. BEV is [B,C,D,H,W] with D the beam depth, and the axes do not "
         "carry the same detail: the lateral penumbra is ~5 mm (2.5 voxels at "
         "2 mm) while depth is build-up plus a smooth exponential. Pooling both "
         "by 16x throws away the lateral structure and keeps depth resolution "
         "nothing needs -- which is precisely why _DilatedRefine3D exists. See "
         "its docstring: it is a full-resolution workaround for this pooling.")
parser.add_argument(
    "--width_growth", type=float, default=2.0,
    help="Channel multiplier per U-Net level. MUST move with --pool_kernels: "
         "cost per level is voxels x channels^2, so a pool that cuts voxels by "
         "v supports growth sqrt(v) at constant cost. 2:2:2 gives v=8 and "
         "break-even 2.83, so the usual doubling is CHEAPER than break-even and "
         "deep levels are nearly free. 2:1:1 gives v=2 and break-even 1.41 -- "
         "leave this at 2.0 there and every level costs twice the one above "
         "instead of half.")
parser.add_argument(
    "--downsample_mode", choices=("maxpool", "strideconv"), default="maxpool",
    help="U-Net encoder downsampling. 'maxpool' reproduces every historical "
         "checkpoint. 'strideconv' uses a learned odd-kernel convolution with "
         "the same per-axis stride as --pool_kernels; for a 2:1:1 schedule it "
         "mixes/downsamples beam depth without touching lateral samples.")
parser.add_argument(
    "--activation_checkpointing", action="store_true",
    help="Checkpoint U-Net encoder, bottleneck and decoder blocks during "
         "training. Recomputes them during backward to reduce activation "
         "memory; evaluation and inference are unchanged.")
parser.add_argument(
    "--sep_levels", type=int, default=0,
    help="Factorise the N FINEST trunk levels into a lateral (1,k,k) conv plus "
         "a depth (kd,1,1) conv. BEV is [B,C,D,H,W] with D the beam depth, so "
         "the two factors are different physics: transverse scatter kernel vs "
         "attenuation/build-up along the ray. 0 = the dense 3x3x3 model every "
         "existing checkpoint uses, key-for-key identical. Only the fine levels "
         "are worth it -- level 0 alone is 54%% of the trunk's multiply-adds "
         "and everything below level 1 is nearly free. 2 is the tested value.")
parser.add_argument(
    "--lateral_kernel", type=int, default=3,
    help="Kernel size of the lateral (1,k,k) factor. Only with --sep_levels "
         "or --separable_refine.")
parser.add_argument(
    "--depth_kernel", type=int, default=5,
    help="Kernel size of the depth (kd,1,1) factor. Cheap to enlarge -- it "
         "costs Cout^2 per step against 9*Cout^2 for the lateral factor -- and "
         "depth is the axis where the physics is genuinely long-range. 5 or 7.")
parser.add_argument(
    "--separable_refine", action="store_true",
    help="Factorise the full-resolution dilated refine branch too. That branch "
         "is 60%% of the corrector's multiply-adds for 1.8%% of its parameters, "
         "so it is where factorisation pays most.")
parser.add_argument(
    "--refine_scale_to_peak", action="store_true",
    help="Bound the refine branch's output to additive_scale_frac of the "
         "control point's dose peak, exactly as the trunk residual already is. "
         "Without it the branch is scale-blind: measured on the e18 "
         "checkpoint, quartering the dose leaves its output at 0.76x while the "
         "trunk residual correctly drops to 0.31x. Changes the output ALGEBRA "
         "and no tensor, so it cannot be inferred from a state_dict.")
parser.add_argument(
    "--refine_depth_dilations", default=None,
    help="Comma-separated depth dilations for the refine branch, one per "
         "layer, e.g. '1,3,9,27' against the lateral default '1,2,4,8'. "
         "Reaching further along depth costs Cout^2 per step instead of "
         "9*Cout^2, so this buys attenuation reach almost free. Requires "
         "--separable_refine.")
parser.add_argument(
    "--no_refine",
    dest="refine",
    action="store_false",
    help="Disable the full-resolution dilated refinement branch. It is ON by "
         "default: it won the architecture sweep, costs 1.6%% of parameters "
         "(22k of 1.36M), and is the only path that sees UN-POOLED input — the "
         "trunk pools 4x, which destroys exactly the sub-voxel edge detail a "
         "1%%/1mm gamma depends on.",
)
parser.set_defaults(refine=True)
parser.add_argument(
    "--residual", action="store_true",
    help="Residual (skip) connection around each U-Net double-conv block — eases training "
         "and helps the corrector model fine residuals on top of an accurate baseline.",
)
parser.add_argument(
    "--no_gamma", action="store_true",
    help="Skip the Level-2 gamma (pymedphys) validation pass. Faster, and avoids the numba "
         "JIT-cache race when many sweep jobs run concurrently. Best checkpoint is then "
         "selected on lvl1 beam MAE (which it is anyway when this is set).",
)
parser.add_argument(
    "--resume",
    default=None,
    help="Continue an interrupted run from a *_last.pt written after an "
         "epoch's training. Unlike --warm_start, which loads WEIGHTS ONLY, "
         "this restores the optimizer moments, the LR schedule position and "
         "the EMA shadow, and skips the epochs already done. Warm-starting a "
         "half-finished run instead restarts the cosine schedule from its "
         "peak and throws away Adam's second-moment estimates, which is not "
         "the same run continued.",
)
parser.add_argument(
    "--ema_decay",
    type=float,
    default=0.0,
    help="Exponential moving average of the model weights for "
         "validation. 0 disables it. ~0.999 smooths the epoch-to-epoch "
         "oscillation seen in the metrics and usually gives a small "
         "free validation gain.",
)
parser.add_argument(
    "--grad_alpha",
    type=float,
    default=1.0,
    help="Weight of the gradient-L1 term (only used with --loss l1_grad).",
)
parser.add_argument(
    "--batched_cp_size",
    type=int,
    default=1,
    help="If >1, training pulls this many CPs from the SAME (patient, beam) "
         "per step and runs them as G segments in one engine forward pass. "
         "The engine already supports that as long as iso_center, "
         "field_size and SID match — which they do within one beam. "
         "Validation stays single-CP so Level-1 metrics are unaffected.",
)
parser.add_argument(
    "--optimizer",
    choices=("adam", "adamw", "sgd"),
    default="adam",
    help="Optimizer. AdamW decouples weight decay from the gradient step "
         "and tends to generalise better on small datasets; SGD+momentum "
         "is a robust fallback.",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Weight decay coefficient. With Adam this acts as additive L2 on "
         "the loss (only useful at very small values); with AdamW it is "
         "applied directly to the weights, which is what you usually want.",
)
parser.add_argument(
    "--sgd_momentum",
    type=float,
    default=0.9,
    help="Momentum used when --optimizer sgd.",
)
parser.add_argument(
    "--lr_schedule",
    choices=("constant", "cosine", "step"),
    default="constant",
    help="LR schedule applied per optimizer step. 'cosine' anneals from "
         "the initial LR down to lr*0.01 over the whole training run; "
         "'step' divides the LR by --step_decay_gamma every "
         "--step_decay_epochs epochs.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="Linear LR warmup applied for this many OPTIMIZER steps before "
         "the chosen schedule takes over. 0 disables warmup.",
)
parser.add_argument(
    "--step_decay_epochs",
    type=int,
    default=150,
    help="Epochs between step-decay drops when --lr_schedule step.",
)
parser.add_argument(
    "--step_decay_gamma",
    type=float,
    default=0.1,
    help="Multiplicative factor at each step-decay drop.",
)
parser.add_argument(
    "--crop_margin_mm",
    type=float,
    default=30.0,
    help="Safety margin (in mm) added to the geometric H-axis crop "
         "computed from MLC jaw_positions. The crop comes from plan "
         "geometry only (no GT dose), so the same code path works at "
         "submission time. 30 mm (~15 slices at 2 mm spacing) covers "
         "penumbra, build-up, and any divergence the box ignores.",
)


_ANATOMY_PREFIXES = {
    "thorax": ["1THB"],
    "abdomen": ["1ABB"],
    "both": ["1THB", "1ABB"],
}


def _resolve_anatomies(arg):
    return _ANATOMY_PREFIXES[arg]


class _TopKByMetric:
    """Bounded cache of the K samples with the highest score on a metric.

    Generalises the earlier MAE-only version. Use it with whatever
    "interestingness" score you want to rank by — e.g. peak-normalised MAE
    so big-field plans don't dominate the ranking just because they
    deposit more dose.
    """

    def __init__(self, k: int):
        self.k = max(0, k)
        self._heap = []          # (score, tiebreak, idx)
        self._payloads = {}      # idx -> payload dict
        self._tiebreak = 0

    def offer(self, score: float, idx: int, payload_factory):
        if self.k == 0:
            return
        if len(self._heap) < self.k:
            self._tiebreak += 1
            heapq.heappush(self._heap, (score, self._tiebreak, idx))
            self._payloads[idx] = payload_factory()
        elif score > self._heap[0][0]:
            _, _, evicted = heapq.heappop(self._heap)
            self._payloads.pop(evicted, None)
            self._tiebreak += 1
            heapq.heappush(self._heap, (score, self._tiebreak, idx))
            self._payloads[idx] = payload_factory()

    def items(self):
        return self._payloads.items()


def _gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on dose gradients along all three spatial axes.

    Plain L1 is happy with a smoothed version of the dose: blurring
    penumbra and density-interface edges costs almost nothing at the
    voxel level, while it costs a lot in terms of clinical correctness.
    Matching ``∇pred`` to ``∇target`` directly penalises that blurring
    and also suppresses high-frequency oscillation in the model's
    correction (the +X / -Y cancelling pattern that drives
    ``mean_absolute_dose_correction`` toward >1).
    """
    spatial = (-3, -2, -1)
    total = pred.new_zeros(())
    for ax in spatial:
        total = total + (pred.diff(dim=ax) - target.diff(dim=ax)).abs().mean()
    return total / len(spatial)


def _corrector_forward_kwargs(args, image, beam, is_batched: bool) -> dict:
    """Extra engine.forward kwargs the v2 feature stack needs.

    MUST be identical in training and validation. It was not: only the training
    loop passed these, so every --v2_features run was VALIDATED on inputs it
    had never seen -- hu_image=None makes engines.build_v2_features substitute
    ``torch.zeros_like(dose_flat)``, collapsing the material channel to
    whatever HU=0 maps to, and open_area_mm2=None sets the aperture-area
    conditioning channel to log1p(0)=0 instead of its real value. Every number
    measured before that fix is biased. Pinned by
    tests/test_v2_support_and_rf.py::test_train_and_val_build_the_same_corrector_inputs.
    """
    if not args.v2_features:
        return {}
    return {
        # HU is required for the material embedding: the HU->density LUT is
        # non-invertible, so density cannot recover material family.
        "hu_image": image.unsqueeze(0),
        "open_area_mm2": _open_area_mm2(beam, is_batched),
    }


def _open_area_mm2(beam, is_batched: bool):
    """Open MLC area per control point, mm^2, for the v2 conditioning vector.

    Aperture area is one of the two axes along which the engine's residual scale
    error was measured to be coherent (ls_scale 0.92 at <705 mm^2 vs 1.00 at
    >5336 mm^2), so the model gets it as a global scalar rather than having to
    infer it from a local patch. Leaf pitch is 5 mm (80 pairs, per
    beam_parameters.json "mlc_leaf_thickness_mm").
    """
    lp = beam.leaf_positions
    if lp.dim() == 2:                      # [N, 2] single CP
        lp = lp.unsqueeze(0)
    width = (lp[..., 1] - lp[..., 0]).clamp_min(0.0)     # [G, N]
    return (width.sum(dim=-1) * 5.0).detach()            # [G]


def _make_loss_fn(args):
    """Return a function (pred, target) -> scalar loss."""
    if args.loss == "l1":
        return lambda pred, target: torch.mean(torch.abs(pred - target))
    if args.loss == "l2":
        return lambda pred, target: torch.mean((pred - target) ** 2)
    if args.loss == "smooth_l1":
        # beta tuned to dose scale (~Gy/CP), values are tiny so beta is small.
        smooth = nn.SmoothL1Loss(beta=1e-2)
        return lambda pred, target: smooth(pred, target)
    if args.loss == "l1_grad":
        alpha = float(args.grad_alpha)
        def loss(pred, target, _a=alpha):
            return torch.mean(torch.abs(pred - target)) + _a * _gradient_l1(pred, target)
        return loss
    if args.loss == "l2_grad":
        alpha = float(args.grad_alpha)
        def loss(pred, target, _a=alpha):
            return torch.mean((pred - target) ** 2) + _a * _gradient_l1(pred, target)
        return loss
    if args.loss == "l1_masked":
        # L1 only where target >= 10% of its peak — matches the
        # challenge's masked MAE so the model doesn't spend capacity on
        # the periphery that nothing scores against. The peak is
        # computed per-CP (per spatial volume) by amax over the last
        # 3 axes, so a small-dose CP isn't drowned out by a big one in
        # the same batch.
        def loss(pred, target):
            peak = target.amax(dim=(-3, -2, -1), keepdim=True)
            m = target >= 0.1 * peak
            diff = (pred - target).abs()
            return diff[m].mean() if bool(m.any()) else diff.mean()
        return loss
    if args.loss in ("l1_banded", "l1_banded_grad"):
        # Dose-band L1, mirroring the challenge's Level-2 stratification so the
        # loss optimises what is actually scored. Bands are fractions of the
        # per-sample target peak (training is per control point, where there is
        # no prescription to normalise by):
        #
        #   high  >= 80%   mid  30-80%   low  10-30%   periphery  2-10%
        #
        # Each band's L1 is normalised by the peak and the bands are averaged
        # with explicit weights, so a band contributes according to its weight
        # rather than to how many voxels it happens to contain. That is the
        # point: the periphery is most of the volume but tiny in dose, so a
        # plain mean L1 ignores it — yet summed over a whole plan those voxels
        # matter. Equally, the high-dose band is few voxels but dominates the
        # clinical result. --band_weights overrides the defaults.
        alpha = float(args.grad_alpha) if args.loss == "l1_banded_grad" else 0.0
        bands = ((0.80, float("inf")), (0.30, 0.80), (0.10, 0.30), (0.02, 0.10))
        if args.band_weights:
            w = [float(v) for v in args.band_weights.split(",")]
            if len(w) != len(bands):
                raise ValueError(f"--band_weights needs {len(bands)} values, got {len(w)}")
        else:
            w = [1.0, 1.0, 1.0, 0.5]
        wsum = sum(w)

        def loss(pred, target, _a=alpha, _b=bands, _w=w, _ws=wsum):
            # Peak-normalised, so a degenerate control point (closed MLC ->
            # all-zero target) must NOT be normalised by its own ~0 peak: that
            # divides by ~1e-9 and the loss explodes to ~5e5, which overflows
            # fp16 under AMP and NaNs the run. Such samples are normalised by 1
            # instead, which leaves rel == 0 everywhere so they fall outside
            # every band and contribute nothing — the correct behaviour, since
            # there is no dose to fit.
            peak = target.amax(dim=(-3, -2, -1), keepdim=True)
            valid = peak > 1e-6
            safe_peak = torch.where(valid, peak, torch.ones_like(peak))
            diff = (pred - target).abs() / safe_peak
            rel = target / safe_peak
            total = torch.zeros((), device=pred.device, dtype=pred.dtype)
            used = 0.0
            for (lo, hi), wt in zip(_b, _w):
                m = (rel >= lo) & (rel < hi)
                if bool(m.any()):
                    total = total + wt * diff[m].mean()
                    used += wt
            out = total / used if used > 0 else (pred - target).abs().mean()
            return out + _a * _gradient_l1(pred, target)
        return loss
    if args.loss == "huber_grad":
        alpha = float(args.grad_alpha)
        smooth = nn.SmoothL1Loss(beta=1e-2)
        def loss(pred, target, _a=alpha):
            return smooth(pred, target) + _a * _gradient_l1(pred, target)
        return loss
    if args.loss == "multiscale":
        alpha = float(args.grad_alpha)

        def loss(pred, target, _a=alpha):
            # avg_pool3d needs 5D [B, C, D, H, W]; train tensors are 4D
            # [B, D, H, W], so add a channel dim for the pooled scales.
            p5 = pred.unsqueeze(1) if pred.dim() == 4 else pred
            t5 = target.unsqueeze(1) if target.dim() == 4 else target
            terms = [(pred - target).abs().mean()]
            for s in (2, 4):
                pp = torch.nn.functional.avg_pool3d(p5, s)
                tt = torch.nn.functional.avg_pool3d(t5, s)
                terms.append((pp - tt).abs().mean())
            ms = sum(terms) / len(terms)
            return ms + _a * _gradient_l1(pred, target)
        return loss
    raise ValueError(f"Unknown loss: {args.loss}")


class _EMA:
    """Exponential moving average of model parameters.

    Updated after every optimizer step; swapped in for validation and
    swapped back out for training. Smooths the epoch-to-epoch metric
    oscillation and usually gives a small free validation gain.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model):
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def swap_out(self, model):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self):
        """The shadow, for resuming. Never call this while swapped IN.

        `swap_in` leaves the shadow untouched and stashes the live weights in
        `_backup`, so the shadow is always the EMA -- but a caller that saved
        `_backup` instead would silently persist the live weights under the EMA
        name, so the assert makes the ordering a hard error rather than a
        subtle one.
        """
        assert not self._backup, (
            "EMA.state_dict() called while swapped in; the caller must "
            "swap_out first or the resumed EMA will be the live weights")
        return {"decay": self.decay,
                "shadow": {n: t.detach().cpu().clone()
                           for n, t in self.shadow.items()}}

    def load_state_dict(self, state, model):
        """Restore the shadow onto `model`'s devices."""
        want = {n for n, p in model.named_parameters() if p.requires_grad}
        got = set(state["shadow"])
        missing, extra = want - got, got - want
        if missing or extra:
            raise ValueError(
                f"EMA shadow does not match the model: {len(missing)} missing "
                f"{sorted(missing)[:3]}, {len(extra)} unexpected "
                f"{sorted(extra)[:3]}. Resuming would average unrelated tensors.")
        ref = {n: p for n, p in model.named_parameters()}
        self.shadow = {n: t.to(device=ref[n].device, dtype=ref[n].dtype)
                       for n, t in state["shadow"].items()}
        self._backup = {}


def _ls_scale(pred, target, eps: float = 1e-12) -> float:
    """Closed-form least-squares scale s minimising ||s*pred - target||².

    Works on either torch tensors or numpy arrays. Returns 1.0 when pred is
    effectively all-zero so we don't divide by ~0.
    """
    if isinstance(pred, torch.Tensor):
        pp = float((pred * pred).sum().item())
        pt = float((pred * target).sum().item())
    else:
        pp = float((pred * pred).sum())
        pt = float((pred * target).sum())
    if pp <= eps:
        return 1.0
    return pt / pp


def _beam_masked_mae_normalized(pred: torch.Tensor, target: torch.Tensor, hi_dose_frac: float = 0.1) -> float:
    """Challenge Level-1 metric 1.1.

    Masked MAE in voxels receiving ≥ hi_dose_frac of the beam's max GT dose,
    divided by the beam's max GT dose. Returns 0 if peak is non-positive.
    """
    peak = float(target.abs().max().item())
    if peak <= 0:
        return 0.0
    diff = (pred - target).abs()
    mask = target > hi_dose_frac * peak
    if not mask.any():
        return 0.0
    return float(diff[mask].mean().item() / peak)


def _idd_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Challenge Level-1 metric 1.2 — IDD curve distance.

    PATIENT SPACE, not BEV. The official evaluator
    (evaluation/doserad2026_evaluator/metrics_beam.py) integrates over the two
    axes other than `beam_axis`, and `beam_axis` defaults to 0 and is never
    overridden -- so the scored curve runs along patient axis 0, which the
    gantry rotates relative to. This used to be fed BEV tensors, which profiles
    along BEAM depth: a different physical quantity that happens to have the
    same units, so the number looked plausible and was not comparable to the
    scored one.

    Sums over the two trailing axes, RMS of the difference of the curves after
    both are normalised by the GT peak. Matches the evaluator to 1e-8.

    Both inputs are 3D patient-space tensors — single beam, no batch.
    """
    pred_idd = pred.sum(dim=(-2, -1)).flatten()
    target_idd = target.sum(dim=(-2, -1)).flatten()
    peak = float(target_idd.abs().max().item())
    if peak <= 0:
        return 0.0
    rms = torch.sqrt(((pred_idd - target_idd) ** 2).mean())
    return float(rms.item() / peak)


def _zero_region_leak(pred_bev: torch.Tensor, target_bev: torch.Tensor) -> tuple:
    """Diagnostic: predicted dose landing where the GT is EXACTLY zero.

    ~80% of a photon target volume is exact zero (measured: 77-82% across
    1ABB006 / 1THB063 / 1THB122). The v2 objective masks to ``t > 0``
    (losses_v2.photon_corrector_loss, no support_mask is passed), so that
    region receives no gradient at all. lv1_idd integrates laterally over ALL
    voxels, so anything deposited there is scored but never trained against.
    Plain ``--loss l1`` (v1) averages over the whole volume and does penalise
    it. These two numbers make that difference measurable:

      leak_mass  spurious predicted mass in the GT-zero region as a fraction
                 of the total true dose.
      leak_idd   RMS over depth of the laterally-integrated leak, normalised
                 by the true IDD peak -- the share of lv1_idd this region
                 alone accounts for.

    Both inputs are 3D BEV tensors (D, H, W), already body-masked.
    """
    total_true = float(target_bev.sum().item())
    if total_true <= 0:
        return 0.0, 0.0
    leak = pred_bev * (target_bev <= 0)
    leak_mass = float(leak.sum().item()) / total_true
    target_idd = target_bev.sum(dim=(-2, -1)).flatten()
    peak = float(target_idd.abs().max().item())
    if peak <= 0:
        return leak_mass, 0.0
    leak_idd = float(torch.sqrt((leak.sum(dim=(-2, -1)).flatten() ** 2).mean()).item())
    return leak_mass, leak_idd / peak


def _stratified_plan_mae(pred: np.ndarray, true: np.ndarray, prescription: float) -> dict:
    """Challenge Level-2 metric 2.1 — stratified plan-level MAE.

    Splits voxels into high (≥80% Rx), mid (30–80% Rx), low (10–30% Rx),
    computes per-stratum MAE / Rx, then returns the unweighted mean across
    strata (per the spec, equal weight to each region).
    """
    if prescription <= 0:
        return {"high": 0.0, "mid": 0.0, "low": 0.0, "combined": 0.0}

    abs_diff = np.abs(pred - true)
    out = {}
    valid_means = []
    bounds = {
        "high": (0.80, np.inf),
        "mid":  (0.30, 0.80),
        "low":  (0.10, 0.30),
    }
    for name, (lo, hi) in bounds.items():
        mask = (true >= lo * prescription) & (true < hi * prescription)
        if mask.any():
            v = float(abs_diff[mask].mean()) / prescription
            out[name] = v
            valid_means.append(v)
        else:
            out[name] = 0.0
    out["combined"] = float(np.mean(valid_means)) if valid_means else 0.0
    return out


def _gamma_pass_rate_3d(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    dd_pct: float = 1.0,
    dta_mm: float = 1.0,
    voxel_mm: float = 2.0,
    eval_thresh_frac: float = 0.10,
    prescription: float | None = None,
    random_subset: int | None = None,
    max_gamma: float = 1.1,
    interp_fraction: int = 10,
    return_map: bool = False,
    **_unused,
):
    """Challenge Level-2 metric 2.2 — 3D local gamma 1%/1mm pass rate (%).

    Delegates to ``pymedphys.gamma`` which is the clinical reference
    implementation: proper sub-voxel reference-dose interpolation, local
    normalisation, and "skip once passed" early termination.

    Knobs that matter for training-time logging cost:
    * ``random_subset``: defaults to None (every evaluated voxel), which is
      what the plan-level metric uses and what matches the official evaluator.
      It exists only for ad-hoc probing; do not use it for reported numbers.
    * ``max_gamma=1.1``: bounds the DTA search. It affects only the reported
      gamma VALUE of failing voxels, NOT the pass/fail decision — anything above
      the cap is still correctly counted as a failure, so the pass rate is
      EXACT. Measured on a deliberately mediocre prediction (44.23% pass):
          max_gamma 2.0 -> 560 s      1.5 -> 323 s      1.1 -> 198 s
          pass rate      44.23%           44.23%            44.23%
      i.e. 2.8x faster for a bit-identical pass rate. This matters most exactly
      when it hurts most: pymedphys only early-terminates on voxels that PASS,
      so gamma gets SLOWER the worse the prediction is, and epoch-0 (corrector
      zero-initialised, prediction = raw engine) is the worst case of the run.
      The cost is a less informative gamma MAP for the plots: every failure now
      saturates at 1.1 instead of showing how badly it failed.
    * previously 2.0: stops searching once gamma exceeds it at a voxel;
      cheap voxels still pass, "definitely failing" voxels stop early.
    * ``interp_fraction=10``: reference dose is interpolated 10× along each
      axis around each evaluation point — enough resolution for 1 mm DTA
      at 2 mm voxel pitch.

    Returns the scalar pass rate by default. With ``return_map=True``,
    returns ``(rate, gamma_array)`` so callers can render the spatial
    failure pattern; ``gamma_array`` has the same shape as ``true`` and
    NaN at voxels that weren't evaluated (below cutoff, outside the
    eval region, or skipped by ``random_subset``).

    Returns NaN (or ``(NaN, None)``) if pymedphys isn't available or
    there are no eval voxels.
    """
    if not _HAS_PYMEDPHYS:
        return (float("nan"), None) if return_map else float("nan")
    if prescription is None:
        prescription = float(np.max(true))
    if prescription <= 0:
        return (float("nan"), None) if return_map else float("nan")

    # pymedphys cares about absolute dose values for the local norm; it
    # treats `lower_percent_dose_cutoff` as a percentage of the max in the
    # reference array. To make that match the challenge's "≥10% of the
    # prescription" we normalise so max == 100 (i.e., units of % of Rx).
    scale = 100.0 / prescription
    ref = (true * scale).astype(np.float32)
    eva = (pred * scale).astype(np.float32)
    if not np.any(ref >= eval_thresh_frac * 100.0):
        return (float("nan"), None) if return_map else float("nan")

    shape = ref.shape
    axes = tuple(np.arange(s, dtype=np.float64) * voxel_mm for s in shape)

    gamma = pymedphys.gamma(
        axes, ref,
        axes, eva,
        dose_percent_threshold=dd_pct,
        distance_mm_threshold=dta_mm,
        lower_percent_dose_cutoff=eval_thresh_frac * 100.0,
        local_gamma=True,
        skip_once_passed=True,
        max_gamma=max_gamma,
        interp_fraction=interp_fraction,
        random_subset=random_subset,
        ram_available=5 * 10 ** 9,   # matches evaluation/.../metrics_plan.py
    )
    valid = ~np.isnan(gamma)
    if not valid.any():
        return (float("nan"), None) if return_map else float("nan")
    rate = float(np.mean(gamma[valid] <= 1.0) * 100.0)
    if return_map:
        return rate, gamma
    return rate


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Raw mean-absolute-error, kept around for loss reporting / ranking.

    The challenge metrics (lv1_beam_mae, lv1_idd, lv2_*) are computed
    elsewhere; this helper now just returns the bare MAE so we have a
    single, unambiguous "training loss" number to print and rank by.
    """
    mae = (pred - target).abs().mean().item()
    return {"mae": mae}


def _resolve_probe_index(index_list, probe_key):
    """Find the dataset position for the probe beam, or the nearest CP if
    the exact one was filtered out (e.g. by ``cp_stride``)."""
    try:
        return index_list.index(probe_key)
    except ValueError:
        pass
    probe_patient, probe_beam, probe_cp = probe_key
    best_idx, best_dist = None, None
    for i, entry in enumerate(index_list):
        patient, beam_index, cp_index = entry[0], entry[1], entry[2]
        if patient != probe_patient or beam_index != probe_beam:
            continue
        d = abs(cp_index - probe_cp)
        if best_dist is None or d < best_dist:
            best_idx, best_dist = i, d
    return best_idx


def _build_payload(ct_volume, target_dose_np, pred_dose_np, iso_center, correction_np, mask_np, idx_key):
    """Pack the lightweight (host-side) arrays the plotter needs.

    NB: do NOT keep references to GPU tensors or to the engine here — those
    would prevent the allocator from reclaiming memory between iterations.
    """
    return {
        "ct_volume": ct_volume,
        "true_dose": target_dose_np,
        "pred_dose": pred_dose_np,
        "iso_center": iso_center,
        "correction": correction_np,
        "body_mask": mask_np,
        "patient_id": idx_key[0],
        "beam_index": idx_key[1],
        "cp_index": idx_key[2],
    }


def _apply_compact_unet_preset(args):
    """Resolve the compact corrector shortcut into ordinary architecture fields.

    Keeping this as a preset rather than a second model class means checkpoint
    loading, baking and serving continue through the existing model_config
    contract.  The full-resolution width is the binding activation-memory term;
    four channels and no refine branch address it directly.
    """
    if not getattr(args, "compact_unet", False):
        return args
    args.base_channels = 4
    args.unet_depth = 3
    args.pool_kernels = "2:1:1,2:1:1,2:1:1"
    args.width_growth = 1.4
    args.downsample_mode = "strideconv"
    args.activation_checkpointing = True
    args.sep_levels = 3
    args.lateral_kernel = 3
    args.depth_kernel = 5
    args.refine = False
    args.attention = "none"
    args.v1_bounded_residual = True
    args.v2_alpha = 0.15
    return args


def _load_fixed_engine_priors(args, device):
    """Load and freeze the focused fluence/heterogeneity engine priors."""
    # Resolve "none" BEFORE the exclusivity check: tiny_fluence_ckpt now has a
    # real default, so a user disabling it with `--tiny_fluence_ckpt none` would
    # otherwise still look like they had asked for it.
    for _a in ("tiny_fluence_ckpt", "lateral_heterogeneity_ckpt"):
        if str(getattr(args, _a, None)).lower() == "none":
            setattr(args, _a, None)

    if args.use_fluence_correction and args.tiny_fluence_ckpt:
        if "--tiny_fluence_ckpt" in sys.argv:
            raise SystemExit(
                "choose --use_fluence_correction or --tiny_fluence_ckpt, not both")
        # Only the DEFAULT prior collided, so honour the explicit legacy request.
        args.tiny_fluence_ckpt = None
        print("[prior] --use_fluence_correction given; the default tiny fluence "
              "prior is disabled for this run", flush=True)

    fluence = None
    fluence_cfg = None
    if args.tiny_fluence_ckpt:
        from fixed_priors import build_tiny_fluence
        payload = torch.load(args.tiny_fluence_ckpt, map_location="cpu",
                             weights_only=False)
        fluence_cfg = payload.get("config") or {}
        fluence = build_tiny_fluence(
            fluence_cfg, payload.get("model_state_dict", payload), device)

    # The heterogeneity prior was fitted on top of TERMA and is meaningless
    # without it. That is a hard error only when it was ASKED for: the ckpt now
    # has a real default, so a TERMA=0 arm would otherwise abort on a prior it
    # never requested. Same rule as the --use_fluence_correction collision.
    if args.lateral_heterogeneity_ckpt and not args.terma:
        if "--lateral_heterogeneity_ckpt" in sys.argv:
            raise SystemExit("--lateral_heterogeneity_ckpt requires --terma")
        args.lateral_heterogeneity_ckpt = None
        print("[prior] --terma is off; the default heterogeneity prior is "
              "disabled for this run", flush=True)

    lateral = None
    lateral_cfg = None
    if args.lateral_heterogeneity_ckpt:
        from fixed_priors import build_tiny_heterogeneity
        payload = torch.load(args.lateral_heterogeneity_ckpt, map_location="cpu",
                             weights_only=False)
        lateral_cfg = payload.get("config") or {}
        try:
            lateral = build_tiny_heterogeneity(
                lateral_cfg, payload.get("model_state_dict", payload), device)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    for name, model in (("fluence", fluence), ("heterogeneity", lateral)):
        if model is not None:
            model.requires_grad_(False).to(device).to(torch.float32).eval()
            print(f"[prior] frozen {name}: "
                  f"{sum(p.numel() for p in model.parameters()):,} params", flush=True)
    return fluence, lateral, fluence_cfg, lateral_cfg


if __name__ == "__main__":
    # Load a local .env (e.g. COMET_API=...) into the environment. A bare
    # .env file does NOT set env vars by itself; python-dotenv (a pydosert
    # dependency) reads it. Searches the CWD upward, so it finds the .env in
    # the dir you ran sbatch from (SLURM sets the job CWD to the submit dir).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    # Comet is optional. Online Experiment() RAISES on a missing api_key, which
    # kills the job ~100 s in, after the container import and CUDA init but
    # before a single step -- an eight-hour ablation lost to a credential. On a
    # cluster node with no key we fall back to OfflineExperiment, which has the
    # same API (set_name/log_metrics/log_image/...) and writes a .zip that
    # `comet upload` can push later. Drop COMET_API into a .env next to this
    # file to get online logging back.
    _comet_key = os.environ.get("COMET_API") or os.environ.get("COMET_API_KEY")
    if _comet_key:
        experiment = Experiment(api_key=_comet_key, project_name="DoseRAD2026")
    else:
        _offline_dir = os.environ.get(
            "COMET_OFFLINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "out", "comet_offline"))
        os.makedirs(_offline_dir, exist_ok=True)
        print(f"[comet] no COMET_API key -- logging OFFLINE to {_offline_dir}",
              flush=True)
        experiment = OfflineExperiment(
            project_name="DoseRAD2026", offline_directory=_offline_dir)
    args = _apply_compact_unet_preset(parser.parse_args())

    # ---- FINALIZED DOSE ENGINE ------------------------------------------
    # These were CLI flags. The engine is settled, and every
    # one of them was a way to run an arm that no longer means anything --
    # or, worse, a wrong combination that trains happily and serves wrong
    # dose. TERMA and lateral scatter model the same disequilibrium and the
    # engine refuses both; a lattice/depth-mode mismatch between training and
    # serving changes the baseline the corrector is a residual OF. Pinning
    # them here keeps every downstream `args.` read working while making the
    # combination unreachable from the command line.
    args.multislab = True
    args.terma, args.terma_c1, args.terma_c2 = True, 0.9, 0.28
    args.lattice_size = 1                 # 1x1 won the {1,3,5} sweep and is cheapest
    args.lattice_depth_mode = "tile_mean"
    args.lattice_tile_chunk = 4           # inert at one tile; the engine still takes it
    args.lateral_scatter = False          # TERMA replaces it
    args.early_bev_crop = True            # bit-identical to uncropped, ~2x cheaper
    args.fluence_to_dose = False
    args.use_fluence_correction = False   # superseded by the frozen tiny fluence prior
    args.mu_eff = None                    # engine default
    args.kernel_size = None
    args.lat_sigma_mm = None
    args.lat_cap_mm = None
    # Corrector-side crop: measured HARMFUL (1ABB045 patient-total 98.31 -> 74.83)
    # and categorical rather than a margin-width effect. --early_bev_crop above
    # is the unrelated, bit-identical physics crop.
    args.bev_crop = False
    args.bev_crop_margin_mm, args.bev_crop_min = 50.0, 32
    args.bev_crop_per_sample = False
    # Retired training paths. --warm_start is NOT retired: fine-tuning from the
    # shipped checkpoint is how arms are run now, so it stays a real flag and
    # no_warm_start is simply its absence.
    args.no_warm_start = not args.warm_start

    # stem_stride: "2" -> 2 (isotropic), "2:1:1" -> (2, 1, 1). Normalised here
    # so the model, the checkpoint's model_config and every rebuilder downstream
    # all see the same object.
    _ss = str(args.stem_stride).strip()
    args.stem_stride = (tuple(int(v) for v in _ss.split(":")) if ":" in _ss
                        else int(_ss))
    args.legacy_val_inputs = False
    args.tta, args.tta_flips = False, "none,w,h,hw"
    args.no_dose_correction = False
    # ---------------------------------------------------------------------

    if args.compact_unet:
        print("[arch] compact_unet: b4 d3 widths~4/6/8/11, "
              "pool=2:1:1 x3, strideconv, sep_levels=3, no refine, "
              "alpha=0.15, activation checkpointing", flush=True)
    if args.seed is not None:
        # Before this existed, weight init and batch order were both
        # unseeded, so an A/B between two arms was confounded by run-to-run
        # variance of the same order as the effects being measured.
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[seed] {args.seed}", flush=True)
    if args.deterministic:
        # Bit-exact reruns need the kernels pinned too, not just the seed.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:            # older torch, or an op with no
            print(f"[deterministic] partial: {exc}", flush=True)  # deterministic impl
        print("[deterministic] cudnn pinned, benchmark off", flush=True)
    amp_dtype = None
    if args.amp:
        if args.amp_dtype == "float16":
            raise SystemExit("--amp_dtype float16 needs a GradScaler, which is "
                             "not implemented. Use bfloat16.")
        amp_dtype = torch.bfloat16
        print(f"[amp] corrector forward in {args.amp_dtype} "
              "(physics stays fp32)", flush=True)

    # VALIDATION runs in fp16, not bfloat16, because that is what the container
    # serves (GC_AMP=fp16). Validating in the training dtype measures a model
    # nobody deploys -- and the same applies to the runtime metric logged from
    # this pass. Training keeps bfloat16: fp16 there would need a GradScaler.
    # Measured cost of fp16 at inference, full 6x3x180 protocol: beam MAE and
    # plan MAE improve, IDD +0.1%, gamma +0.01 pp. See
    # (fp16 at inference was measured neutral-to-better; see the Dockerfile.)
    val_amp_dtype = torch.float16 if args.amp else None
    if args.run_tag:
        experiment.set_name(args.run_tag)
    # WHICH MACHINE. runtime_per_beam is wall clock, and SLURM puts each job on
    # whatever node is free, so two arms can differ by more in hardware than in
    # anything under test. Without this, a runtime difference between runs is
    # uninterpretable. Logged as `other` so it shows on the run without
    # polluting the hyperparameter diff.
    import socket as _socket
    _hw = {
        "host": _socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "-"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", "-"),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "gpu_count": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "torch": torch.__version__,
    }
    for _k, _v in _hw.items():
        experiment.log_other(_k, _v)
    print(f"[hw] {_hw['gpu']} on {_hw['host']} "
          f"(job {_hw['slurm_job_id']}, nodes {_hw['slurm_nodelist']})", flush=True)
    data_path = args.data_path or default_data_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    treatment_modality = args.treatment
    modality = [m.strip() for m in args.modality.split(",") if m.strip()]
    if len(modality) > 1:
        print(f"[train] modalities: {modality} -- the corrector sees real CT and "
              "synthetic CT, which is what it is deployed on", flush=True)
    learning_rate = args.lrate
    l1_norm_alpha = args.l1_alpha
    accumulation_steps = args.batch_size
    anatomies = _resolve_anatomies(args.anatomies)
    # Lateral-scatter sigma: scalar, comma-separated per-bin list, or the fitted default.
    if args.lat_sigma_mm is None:
        args.lat_sigma_mm = MULTISLAB_DEFAULTS["lat_sigma_mm"]
    elif "," in str(args.lat_sigma_mm):
        args.lat_sigma_mm = [float(v) for v in str(args.lat_sigma_mm).split(",")]
    else:
        args.lat_sigma_mm = float(args.lat_sigma_mm)
    if args.lat_cap_mm is None:
        args.lat_cap_mm = MULTISLAB_DEFAULTS["lat_cap_mm"]
    if args.mu_eff is None:
        args.mu_eff = MULTISLAB_DEFAULTS["mu_eff"]
    print(f"[engine] multislab={args.multislab} mu_eff={args.mu_eff} "
          f"lateral_scatter={args.lateral_scatter} lat_sigma_mm={args.lat_sigma_mm} "
          f"lat_cap_mm={args.lat_cap_mm}")
    patient_filter = [args.patient] if args.patient else None
    beam_indices = list(range(args.num_beams))
    cp_indices = list(range(0, 180, args.cp_stride))[:args.num_cps]
    # Validation beams: default to the same beams as training, but --val_num_beams lets us
    # validate on fewer beams (e.g. just beam 0) to cut the (slow) validation time.
    val_beam_indices = list(range(args.val_num_beams)) if args.val_num_beams else beam_indices

    dataset = BeamDataset(
        data_path, "training", modality,
        anatomies=anatomies, patients=patient_filter,
        beam_indices=beam_indices, cp_indices=cp_indices,
        crop_margin_mm=args.crop_margin_mm,
    )
    # Validation always uses every CP so the plan reconstruction is the
    # full clinical plan, not a strided approximation. cp_stride only
    # subsamples training.
    # Validation covers BOTH arms in one pass: the corrector is trained on real
    # CT, but it is deployed on CT and on MR-derived synthetic CT, so both must
    # be scored every epoch. One dataset, modality carried per sample.
    val_modalities = [m.strip() for m in args.val_modalities.split(",") if m.strip()]
    print(f"[val] modalities: {val_modalities}")
    val_dataset = BeamDataset(
        data_path, "validating", val_modalities,
        anatomies=anatomies, patients=patient_filter,
        beam_indices=val_beam_indices,
        cp_indices=list(range(0, 180, max(1, args.val_cp_stride))),
        crop_margin_mm=args.crop_margin_mm,
    )
    kernel_size = args.kernel_size if args.kernel_size is not None else DEFAULT_KERNEL_SIZE
    num_epochs = args.num_epochs

    # machine_config is the calibrated engine baseline imported from
    # configs.py (TPR / energy / penumbra / head-scatter / mlc all baked
    # into baseline_machine_config.json + _DEFAULT_PARAMS). No per-run
    # overrides any more — calibration is finished.
    print(f"MachineConfig (calibrated): tpr_20_10={machine_config.tpr_20_10} "
          f"E_MeV={machine_config.mean_photon_energy_MeV} "
          f"penumbra={machine_config.penumbra_fwhm} "
          f"mlc_t={machine_config.mlc_transmission} "
          f"h_s amp={machine_config.head_scatter_amplitude} "
          f"sigma={machine_config.head_scatter_sigma} "
          f"kernel_size={kernel_size}")
    _worker_kw = {"num_workers": args.num_workers, "collate_fn": same_beam_collate}
    if args.num_workers > 0:
        _worker_kw["persistent_workers"] = True   # reuse workers across epochs (no respawn RAM spike)
        _worker_kw["prefetch_factor"] = 2
    if args.seed is not None:
        # Workers are separate processes with their own RNGs, and the sampler
        # draws from the loader's generator -- seeding the parent alone leaves
        # batch order random.
        def _seed_worker(worker_id: int, _base: int = args.seed) -> None:
            import random as _r
            s = _base + worker_id
            _r.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
        _gen = torch.Generator()
        _gen.manual_seed(args.seed)
        _worker_kw["generator"] = _gen
        if args.num_workers > 0:
            _worker_kw["worker_init_fn"] = _seed_worker
    if args.batched_cp_size > 1:
        train_batch_sampler = SameBeamBatchSampler(
            dataset.index_list, batch_size=args.batched_cp_size,
            shuffle=True, drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=train_batch_sampler,
            **_worker_kw,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            **_worker_kw,
        )
    # Real batching also for validation when --batched_cp_size > 1: G
    # same-beam CPs per engine forward, with per-CP metrics still
    # computed by unpacking the G dim inside the loop.
    if args.batched_cp_size > 1:
        val_batch_sampler = SameBeamBatchSampler(
            val_dataset.index_list, batch_size=args.batched_cp_size,
            shuffle=False, drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            **_worker_kw,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            **_worker_kw,
        )

    fixed_fluence_cfg = fixed_lateral_cfg = None
    lateral_heterogeneity_model = None
    if args.tiny_fluence_ckpt or args.lateral_heterogeneity_ckpt:
        (fluence_correction_model, lateral_heterogeneity_model,
         fixed_fluence_cfg, fixed_lateral_cfg) = _load_fixed_engine_priors(args, device)
    else:
        fluence_correction_model = (FluenceCorrectionModel().to(device).to(torch.float32)
                                    if args.use_fluence_correction else None)
    corrector_in_channels = 2 + (1 if args.multislab else 0) + (1 if args.use_source_distance else 0)
    v2_feature_cfg = None
    # A non-v2 arch can still take the v2 feature stack: raw dose stays channel
    # 0 (v1's gain head multiplies it and the engine adds the result back in
    # physical units), then the v2 channels follow.
    if args.v2_features:
        from features_v2 import FeatureConfig as _FC
        v2_feature_cfg = _FC(cond_mode=args.v2_cond_mode,
                             feature_set=args.feature_set,
                             cond_region=args.cond_region,
                             crop_relative_position=args.bev_crop)
        corrector_in_channels = 1 + v2_feature_cfg.n_scalar_channels()
        print(f"[v2feat] unet gets {corrector_in_channels} channels "
              f"(raw dose + {v2_feature_cfg.n_scalar_channels()} "
              f"{args.feature_set} channels, cond={args.v2_cond_mode})")
    material_embed_dim = int(args.v1_material_embed_dim)
    if material_embed_dim > 0 and not args.v2_features:
        raise SystemExit(
            "--v1_material_embed_dim needs --v2_features: the material id is "
            "produced by the v2 feature builder, and without it the engine has "
            "nothing to pass.")
    if args.bev_crop and not args.v2_features:
        raise SystemExit(
            "--bev_crop needs --v2_features: the legacy channel path builds "
            "its corrector inputs from full-volume tensors that the crop does "
            "not slice.")
    if args.bev_crop and args.feature_set not in ("v3", "v4"):
        # depth_index is linspace(0,1) over the BEV box, so under a crop it
        # measures position within the CROP -- and the crop size varies per
        # beam, so the channel would mean something different on every control
        # point. v3 drops exactly that channel, and v4 inherits the drop.
        # Pinned by
        # tests/test_bev_crop.py::test_v2_depth_index_is_why_the_crop_requires_v3.
        raise SystemExit(
            f"--bev_crop requires --feature_set v3 or v4 (got "
            f"{args.feature_set!r}): the v2 stack carries depth_index, which "
            f"is crop-relative, so cropping silently redefines it per beam.")
    if args.bev_crop_per_sample and not args.bev_crop:
        raise SystemExit(
            "--bev_crop_per_sample requires --bev_crop: it changes how the "
            "box is cut, and without a crop there is no box.")
    # --early_bev_crop no longer requires --bev_crop. They are different things
    # and only one of them changes the dose:
    #
    #   --bev_crop       crops what the CORRECTOR SEES. It changes the model's
    #                    input distribution, and a patient-total bisect
    #                    attributed the abdomen regression to it: 1ABB045 went
    #                    98.31 (no crop) -> 74.83 (crop only) -> 72.09 (crop
    #                    plus every later change). Margin size was almost
    #                    irrelevant (50 -> 30 mm cost 2.7 points against 23.5
    #                    for cropping at all), so it is categorical, not a
    #                    context-width effect.
    #
    #   --early_bev_crop crops the PHYSICS WORK VOLUME before the pencil-beam
    #                    convolution and scatters the result back. Measured
    #                    bit-identical to the uncropped engine -- max absolute
    #                    difference exactly 0.0 -- because the retained halo is
    #                    derived from the configured kernel and scatter sigmas.
    #                    It is a pure speed optimisation and cannot change what
    #                    the corrector is fitted to.
    #
    # Coupling them forced a choice between "corrector sees the full volume"
    # and "the physics is affordable", which is exactly the wrong trade with a
    # ray lattice: the lattice multiplies the convolution by N^2, and the early
    # crop nearly halves it (3x3: 74.6 -> 45.2 ms per control point).
    if args.v1_deep_supervision and not args.v2_loss:
        # The deep term lives only in losses_v2. Without it the aux heads would
        # still be built and run every step, costing compute and appearing in
        # the state_dict, while contributing nothing to the gradient -- the
        # mirror image of the bug this feature exists to fix.
        raise SystemExit(
            "--v1_deep_supervision needs --v2_loss: the deep-supervision term "
            "exists only in the v2 composite loss, so without it the aux heads "
            "would train nothing.")
    _pool_kernels = None
    if args.pool_kernels:
        _pool_kernels = [tuple(int(v) for v in grp.split(":"))
                         for grp in args.pool_kernels.split(",") if grp.strip()]
        if any(len(k) != 3 for k in _pool_kernels):
            raise SystemExit(f"--pool_kernels needs D:H:W triples, got "
                             f"{args.pool_kernels!r}")
        if len(_pool_kernels) != args.unet_depth:
            raise SystemExit(
                f"--pool_kernels has {len(_pool_kernels)} entries but "
                f"--unet_depth is {args.unet_depth}; one pool per level.")
    _refine_depth_dilations = None
    if args.refine_depth_dilations:
        if not args.separable_refine:
            raise SystemExit(
                "--refine_depth_dilations needs --separable_refine: a dense "
                "3x3x3 conv has one dilation for all three axes, so the depth "
                "schedule would be silently ignored.")
        _refine_depth_dilations = tuple(
            int(v) for v in args.refine_depth_dilations.split(",") if v.strip())
    if True:
        dose_correction_model = DoseCorrectionModel(
            in_channels=corrector_in_channels,
            base_channels=args.base_channels,
            depth=args.unet_depth,
            use_gain=not args.no_gain_head,
            norm=args.v1_norm,
            bounded_residual=args.v1_bounded_residual,
            additive_scale_frac=args.v2_alpha,
            attention=args.attention,
            refine=args.refine,
            residual=args.residual,
            material_embedding_dim=material_embed_dim,
            deep_supervision=args.v1_deep_supervision,
            pool_kernels=_pool_kernels,
            width_growth=args.width_growth,
            downsample_mode=args.downsample_mode,
            activation_checkpointing=bool(args.activation_checkpointing),
            sep_levels=args.sep_levels,
            lateral_kernel=args.lateral_kernel,
            depth_kernel=args.depth_kernel,
            separable_refine=bool(args.separable_refine),
            refine_scale_to_peak=bool(args.refine_scale_to_peak),
            refine_hidden=args.refine_hidden,
            refine_depth_dilations=_refine_depth_dilations,
            stem_stride=args.stem_stride,
            refine_mode=args.refine_mode,
        )
        if args.sep_levels or args.separable_refine:
            n_par = sum(p.numel() for p in dose_correction_model.parameters())
            print(f"[arch] separable: sep_levels={args.sep_levels} "
                  f"lateral={args.lateral_kernel} depth={args.depth_kernel} "
                  f"refine={'sep' if args.separable_refine else 'dense'} "
                  f"h{args.refine_hidden} -> {n_par:,} params")
        if material_embed_dim > 0:
            print(f"[v3] material embedding: 86 families -> "
                  f"{material_embed_dim} channels "
                  f"({corrector_in_channels + material_embed_dim} at the stem)")
        if args.v1_deep_supervision and dose_correction_model is not None:
            n_aux = len(dose_correction_model.aux_heads or ())
            print(f"[v3] deep supervision: {n_aux} aux heads, w_deep="
                  f"{args.v2_w_deep}")
    if args.no_dose_correction:
        if not args.use_fluence_correction:
            raise ValueError(
                "--no_dose_correction with no --use_fluence_correction leaves "
                "nothing to train: the engine would be a fixed function.")
        dose_correction_model = None
        print("[corrector] 3D dose corrector DISABLED -- training the fluence "
              "corrector alone against the bare engine.")
    else:
        dose_correction_model = dose_correction_model.to(device).to(torch.float32)

    # Everything needed to REBUILD this model, carried with the weights.
    #
    # Three of these leave no recoverable trace in the state_dict:
    # v1_bounded_residual changes the output ALGEBRA and no tensor at all;
    # feature_set changes which 12 channels the builder emits without changing
    # the count; cond_region redefines two of those channels while leaving the
    # count and the names alone; and v2_cond_mode changes the count. A shape-matching
    # loader therefore accepts the wrong configuration and returns a wrong dose
    # rather than an error -- which is how a multislab corrector once shipped
    # bake_model.py copies this forward.
    corrector_model_config = {
        "arch": "unet",
        "in_channels": corrector_in_channels,
        "base_channels": args.base_channels,
        "depth": args.unet_depth,
        "norm": args.v1_norm,
        "bounded_residual": bool(args.v1_bounded_residual),
        "refine": bool(args.refine),
        "compact_unet": bool(args.compact_unet),
        # Architecture, not a knob: sep_levels/lateral_kernel/depth_kernel and
        # the refine flags change the state_dict KEYS, so a mismatch fails the
        # load loudly rather than silently -- but refine_hidden alone changes
        # only shapes, and refine_depth_dilations changes NOTHING in the
        # state_dict at all. That last one is the dangerous member: a model
        # served with the wrong depth dilations loads clean and returns a
        # plausible wrong dose.
        # Both change the state_dict (widths and skip shapes), so a mismatch
        # fails the load loudly rather than silently.
        "pool_kernels": ([list(k) for k in _pool_kernels] if _pool_kernels else None),
        "width_growth": float(args.width_growth),
        "downsample_mode": args.downsample_mode,
        "activation_checkpointing": bool(args.activation_checkpointing),
        "sep_levels": int(args.sep_levels),
        "lateral_kernel": int(args.lateral_kernel),
        "depth_kernel": int(args.depth_kernel),
        "separable_refine": bool(args.separable_refine),
        "refine_scale_to_peak": bool(args.refine_scale_to_peak),
        "refine_hidden": int(args.refine_hidden),
        "stem_stride": (list(args.stem_stride)
                        if isinstance(args.stem_stride, tuple)
                        else int(args.stem_stride)),
        "refine_mode": args.refine_mode,
        "refine_depth_dilations": (list(_refine_depth_dilations)
                                   if _refine_depth_dilations else None),
        "residual": bool(args.residual),
        "attention": args.attention,
        "use_gain": not args.no_gain_head,
        "additive_scale_frac": float(args.v2_alpha),
        "v2_features": bool(args.v2_features),
        "v2_cond_mode": args.v2_cond_mode,
        "feature_set": args.feature_set if args.v2_features else None,
        "cond_region": args.cond_region if args.v2_features else None,
        "material_embedding_dim": material_embed_dim,
        "deep_supervision": bool(args.v1_deep_supervision),
        "engine": "multislab",
        # Engine physics, not architecture -- the weights are identical either
        # way -- but the corrector predicts the residual of ONE baseline, and
        # serving it on a different one measured plan gamma 93.3 -> 63.0. So
        # these travel with the weights like bev_crop does.
        "lattice_size": int(args.lattice_size),
        "lattice_depth_mode": args.lattice_depth_mode,
        "terma": bool(args.terma),
        "terma_c1": (float(args.terma_c1) if args.terma else None),
        "terma_c2_per_mm": (float(args.terma_c2) if args.terma else None),
        "fixed_fluence_config": fixed_fluence_cfg,
        "fixed_lateral_heterogeneity_config": fixed_lateral_cfg,
        # Not an architecture field -- the trunk is fully convolutional and the
        # weights are identical either way -- but it changes what the model was
        # trained to see, so inference has to reproduce it or the corrector
        # meets a distribution it never saw.
        "bev_crop": bool(args.bev_crop),
        "bev_crop_margin_mm": float(args.bev_crop_margin_mm),
        "bev_crop_min": int(args.bev_crop_min),
        "early_bev_crop": bool(args.early_bev_crop),
        # Changes what lat_h/lat_w MEAN without changing the channel count, so
        # nothing downstream can detect a mismatch. Recorded, like the rest.
        "bev_crop_per_sample": bool(args.bev_crop_per_sample),
    }

    # Warm-start from the modality-specific best ckpt (unless explicitly
    # disabled). Done before EMA construction so the EMA shadow starts
    # from the loaded weights, not the random init that was just
    # overwritten. Optimizer state is intentionally NOT restored — these
    # are fresh runs that just inherit good initial weights.
    if args.fluence_to_dose and not args.no_warm_start:
        print("[warm-start] disabled: --fluence_to_dose trains the CNN from random "
              "init (input is the fluence volume, not the dose-correction baseline).")
    if not args.no_warm_start and not args.fluence_to_dose and args.warm_start:
        init_path = args.warm_start
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f"[warm-start] checkpoint not found: {init_path}. Provide the file, "
                f"fix --warm_start, or pass --no_warm_start for random init.")
        if dose_correction_model is None:
            raise ValueError(
                "--warm_start needs a dose corrector; --no_dose_correction "
                "removed it. Pass --no_warm_start.")
        warm_start_from(dose_correction_model, init_path)

    # EMA follows whatever is being trained. With --no_dose_correction that is
    # the fluence corrector, and shadowing the (absent) dose model would either
    # crash or silently average nothing.
    _ema_target = (dose_correction_model if dose_correction_model is not None
                   else fluence_correction_model)
    ema = _EMA(_ema_target, args.ema_decay) if args.ema_decay > 0 else None

    experiment.log_parameters({
        "treatment_modality": treatment_modality,
        "modality": ",".join(modality),
        "kernel_size": kernel_size,
        "learning_rate": learning_rate,
        "l1_norm_alpha": l1_norm_alpha,
        "batch_size": accumulation_steps,
        "anatomies": ",".join(anatomies),
        "patient_filter": args.patient or "all",
        "num_beams": args.num_beams,
        "num_cps": args.num_cps,
        "arch": "unet",
        "base_channels": args.base_channels,
        "unet_depth": args.unet_depth,
        "use_gain_head": not args.no_gain_head,
        "attention": args.attention,
        "refine": args.refine,
        "use_source_distance": args.use_source_distance,
        "feature_set": args.feature_set if args.v2_features else None,
        "cond_region": args.cond_region if args.v2_features else None,
        "material_embedding_dim": args.v1_material_embed_dim,
        "deep_supervision": args.v1_deep_supervision,
        "w_deep": args.v2_w_deep if args.v1_deep_supervision else 0.0,
        "ema_decay": args.ema_decay,
        "max_train_steps": args.max_train_steps if args.max_train_steps is not None else -1,
        "loss_fn": args.loss,
        "grad_alpha": args.grad_alpha if args.loss in ("l1_grad", "l2_grad", "huber_grad", "multiscale") else 0.0,
        "batched_cp_size": args.batched_cp_size,
        "optimizer": args.optimizer,
        "adam_eps": float(args.adam_eps),
        "weight_decay": args.weight_decay,
        "sgd_momentum": args.sgd_momentum if args.optimizer == "sgd" else 0.0,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "step_decay_epochs": args.step_decay_epochs,
        "step_decay_gamma": args.step_decay_gamma,
        "crop_margin_mm": args.crop_margin_mm,
        "kernel_size": kernel_size,
        "machine_tpr_20_10": float(machine_config.tpr_20_10),
        "machine_mean_photon_energy_MeV": float(machine_config.mean_photon_energy_MeV),
        "machine_mlc_transmission": float(getattr(machine_config, "mlc_transmission", 0.0)),
        "machine_penumbra_fwhm_0": float(machine_config.penumbra_fwhm[0]) if machine_config.penumbra_fwhm else 0.0,
        "machine_head_scatter_amp_0": float(machine_config.head_scatter_amplitude[0]) if machine_config.head_scatter_amplitude else 0.0,
        "machine_head_scatter_sigma_0": float(machine_config.head_scatter_sigma[0]) if machine_config.head_scatter_sigma else 0.0,
        "machine_head_scatter_sigma_1": float(machine_config.head_scatter_sigma[1]) if machine_config.head_scatter_sigma else 0.0,
        "machine_head_scatter_ssd_mm": float(machine_config.head_scatter_ssd_mm),
        "machine_dlg_mm": float(machine_config.dlg_mm) if getattr(machine_config, "dlg_mm", None) is not None else -1.0,
        "machine_has_profile_corrections": getattr(machine_config, "profile_corrections", None) is not None,
        "machine_has_output_factors": getattr(machine_config, "output_factors", None) is not None,
        "use_fluence_correction": args.use_fluence_correction,
        "dose_correction_model_params": sum(p.numel() for p in dose_correction_model.parameters()) if dose_correction_model is not None else 0,
        "fluence_correction_model_params": sum(p.numel() for p in fluence_correction_model.parameters()) if fluence_correction_model is not None else 0,
    })
    if args.tta:
        if args.v2_features:
            raise SystemExit("--tta needs unsigned scalar inputs; the v2 "
                             "feature stack carries signed lateral position "
                             "channels that a plain mirror would corrupt.")
        if args.v1_deep_supervision:
            # LateralTTA averages tensors. With aux heads the model returns a
            # dict, which it would try to add to a tensor.
            raise SystemExit("--tta cannot wrap a --v1_deep_supervision model: "
                             "LateralTTA is tensor-in/tensor-out.")
        from engines import LateralTTA
        _map = {"none": (), "w": (-1,), "h": (-2,), "hw": (-2, -1)}
        _sel = tuple(_map[t.strip()] for t in args.tta_flips.split(","))
        dose_correction_model = LateralTTA(dose_correction_model, flips=_sel).to(device)
        print("[tta] validation averages over 4 lateral mirrors "
              f"({args.tta_flips})")

    criterion = _make_loss_fn(args)
    v2_loss_cfg = None
    use_v2_loss = bool(args.v2_loss)
    if use_v2_loss:
        from losses_v2 import LossConfig, photon_corrector_loss
        # support_mode and w_idd are left at their LossConfig defaults
        # ("positive", 0.0). Both were exposed as CLI flags and both
        # hypotheses were tested and rejected -- support_mode="all" is a
        # no-op, and the profile/IDD term is harmful -- so the knobs live in
        # losses_v2.LossConfig for anyone who wants to re-open them, and no
        # longer in the command line where they can silently override the
        # measured default.
        v2_loss_cfg = LossConfig(w_high10=args.v2_w_high10,
                                 w_deep=args.v2_w_deep,
                                 w_strata=args.v2_w_strata,
                                 w_rel=args.v2_w_rel,
                                 w_logw=args.v2_w_logw,
                                 w_idd=args.v2_w_idd,
                                 w_mse=args.v2_w_mse)
        print(f"[v2] loss: support_mae({v2_loss_cfg.support_mode}) "
              f"+ {v2_loss_cfg.w_high10}*high10 "
              f"+ {v2_loss_cfg.w_deep}*deep")

    # Whatever is actually trainable this run. Previously hard-wired to the
    # dose corrector, so --use_fluence_correction built a fluence network the
    # optimiser never saw -- the second of two independent reasons it could
    # not learn (the first being the no_grad in the engine).
    _trainable = [m for m in (dose_correction_model, fluence_correction_model)
                  if m is not None]
    _trainable_params = [p for m in _trainable for p in m.parameters()]
    if not _trainable_params:
        raise ValueError("nothing to optimise: no dose or fluence corrector")
    print(f"[optim] optimising {sum(p.numel() for p in _trainable_params):,} "
          f"parameters across {len(_trainable)} module(s)")

    if args.optimizer == "adam":
        optimizer = optim.Adam(
            _trainable_params, eps=args.adam_eps,
            lr=learning_rate, weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            _trainable_params,
            lr=learning_rate, weight_decay=args.weight_decay, eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            _trainable_params,
            lr=learning_rate, momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # LR schedule: one LambdaLR that handles warmup + the chosen decay
    # shape. Stepped once per OPTIMIZER step (i.e. once per gradient
    # accumulation cycle), not once per data sample.
    steps_per_epoch = (args.max_train_steps if args.max_train_steps is not None
                       else len(loader))
    optim_steps_per_epoch = max(1, steps_per_epoch // accumulation_steps)
    total_optim_steps = max(1, num_epochs * optim_steps_per_epoch)
    warmup = max(0, args.warmup_steps)
    step_decay_period = max(1, args.step_decay_epochs * optim_steps_per_epoch)

    def _lr_lambda(step):
        # step counts OPTIMIZER steps from 0.
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if args.lr_schedule == "constant":
            return 1.0
        if args.lr_schedule == "cosine":
            prog = (step - warmup) / max(1, total_optim_steps - warmup)
            prog = min(max(prog, 0.0), 1.0)
            # Decay from 1.0 down to 0.01 (i.e. lr*0.01 at the end).
            return 0.5 * (1.0 + math.cos(math.pi * prog)) * 0.99 + 0.01
        if args.lr_schedule == "step":
            n_decays = max(0, (step - warmup) // step_decay_period)
            return float(args.step_decay_gamma ** n_decays)
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    print(f"Optimizer={args.optimizer} wd={args.weight_decay} "
          f"schedule={args.lr_schedule} warmup={warmup} optim-steps "
          f"~{total_optim_steps} ({optim_steps_per_epoch}/epoch)")

    # ---- Resume ----------------------------------------------------------
    # Restores the full training state, not just the weights. Done here, after
    # the optimizer and scheduler exist and before the epoch loop, so the
    # schedule resumes at the step it stopped at rather than back at the
    # cosine peak.
    start_epoch = 0
    resumed_best_lv1 = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise SystemExit(f"--resume: no such checkpoint: {args.resume}")
        _r = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("optimizer_state_dict", "scheduler_state_dict"):
            if key not in _r:
                raise SystemExit(
                    f"--resume: {args.resume} has no {key}. Only a *_last.pt "
                    f"carries the training state; the plain and _best files "
                    f"hold weights alone, and resuming from one would restart "
                    f"the schedule and lose the optimiser moments. Use "
                    f"--warm_start for those.")
        if dose_correction_model is not None and _r.get("dose_model_state_dict"):
            dose_correction_model.load_state_dict(_r["dose_model_state_dict"])
        optimizer.load_state_dict(_r["optimizer_state_dict"])
        scheduler.load_state_dict(_r["scheduler_state_dict"])
        # The EMA is part of the training state too. Resuming without it would
        # restart the average from the live weights, and the first validation
        # after the restart would report something neither run produced.
        if ema is not None:
            if not _r.get("ema_state_dict"):
                raise SystemExit(
                    f"--resume: --ema_decay {args.ema_decay} is on but "
                    f"{args.resume} carries no EMA shadow (it was written by a "
                    f"run without EMA). Continue without --ema_decay, or start "
                    f"a fresh run.")
            ema.load_state_dict(_r["ema_state_dict"], dose_correction_model)
        elif _r.get("ema_state_dict"):
            print("[resume] WARNING: checkpoint has an EMA shadow but "
                  "--ema_decay is 0; the average is being discarded.")
        start_epoch = int(_r.get("epochs_trained", _r.get("epoch", 0)))
        resumed_best_lv1 = _r.get("best_val_lv1_beam_mae")
        print(f"[resume] {args.resume}: {start_epoch} epochs trained, "
              f"scheduler at step {scheduler.last_epoch}, "
              f"lr={optimizer.param_groups[0]['lr']:.3e}")
        if start_epoch >= num_epochs:
            raise SystemExit(
                f"--resume: checkpoint already has {start_epoch} epochs and "
                f"--num_epochs is {num_epochs}; nothing left to do. Raise "
                f"--num_epochs to continue.")

    # The "probe" beam — kept stable across epochs so progress over training is
    # comparable. If the exact CP was filtered out (cp_stride>1) we fall back
    # to the nearest CP for the same patient+beam so the visualisation is
    # always present.
    probe_key = ('1THB043', 0, 15)
    probe_index = _resolve_probe_index(val_loader.dataset.index_list, probe_key)
    if probe_index is not None:
        actual = val_loader.dataset.index_list[probe_index]
        if actual != probe_key:
            print(
                f"Probe {probe_key} not in dataset (cp_stride filter); "
                f"using nearest CP: {actual}"
            )

    os.makedirs("out", exist_ok=True)

    # Name checkpoints after the Comet experiment so concurrent sweep jobs
    # don't clobber a single shared file and so a finished run can be warm-
    # started later instead of retrained from scratch. get_name() is the
    # human-readable Comet name (e.g. "witty_panda_1234"); fall back to the
    # key, then a constant, and sanitise for the filesystem.
    _raw_run_name = args.run_tag or experiment.get_name() or experiment.get_key() or "run"
    run_name = re.sub(r"[^A-Za-z0-9_.-]", "_", _raw_run_name)
    ckpt_path = f"out/{run_name}.pt"
    ckpt_best_path = f"out/{run_name}_best.pt"
    # Written AFTER each epoch's training, unlike the two above, which are
    # written during the validation that runs at the TOP of an epoch. Without
    # it the final epoch's training is simply lost: the loop trains, the range
    # ends, and the process exits before the next validation would have saved
    # anything. It also carries the optimiser, scheduler and EMA state, which
    # is what --resume needs and what the other two deliberately omit.
    ckpt_last_path = f"out/{run_name}_last.pt"
    print(f"Checkpoints -> {ckpt_path} / {ckpt_best_path} / {ckpt_last_path}")

    if not _HAS_PYMEDPHYS:
        print("WARNING: pymedphys not installed — Level-2 gamma will be NaN. "
              "Run `pip install pymedphys` to enable it.")

    # Best checkpoint is selected by mean Level-2 gamma (1%/1mm) pass
    # rate — the most clinically meaningful score we compute. Higher is
    # better, so we track the running max; -inf means "no valid gamma
    # seen yet" (first epoch with finite gamma always wins).
    best_val_gamma = -float("inf")
    best_val_lv1 = float("inf")   # lvl1 beam MAE (lower = better) for best-ckpt selection
    if resumed_best_lv1 is not None and resumed_best_lv1 == resumed_best_lv1:
        # Carry the bar forward, or the first validation after a resume
        # overwrites _best.pt with whatever it happens to score.
        best_val_lv1 = float(resumed_best_lv1)
        print(f"[resume] best val_lv1_beam_mae so far: {best_val_lv1:.4f}")
    for epoch in range(start_epoch, num_epochs):
        if dose_correction_model is not None:
            dose_correction_model.eval()
        if fluence_correction_model is not None:
            fluence_correction_model.eval()
        if lateral_heterogeneity_model is not None:
            lateral_heterogeneity_model.eval()
        # Validate with the EMA weights (smoother, less oscillation).
        if ema is not None:
            ema.swap_in(_ema_target)

        val_loss_total = 0.0
        num_batches = 0
        training_loss = []
        mean_absolute_correction = []
        mean_absolute_dose_correction = []
        # Wall-clock prediction time across the whole val pass. We sum
        # per-batch (engine_init + forward + sync) times here and divide
        # by total CPs at the end — the resulting runtime_per_beam
        # therefore includes the engine construction overhead that was
        # previously excluded (challenge metric is per-beam inference
        # cost; engine setup is part of inference, not bookkeeping).
        # Anything below the synced forward (BEV rotate, metrics, plots,
        # plan accumulation, gamma) is evaluation and stays outside.
        # Split into the two terms the leaderboard actually bills. At submission
        # gc_inference builds ONE engine per beam and reuses it across all
        # `num_cps` control points; validation rebuilds it every
        # batched_cp_size batch. Amortising a full engine build over 3 CPs
        # instead of 180 inflated runtime_per_beam by up to 60x of the init
        # term, and made the metric sensitive to build-time noise that the real
        # submission never pays. Timed separately, recomposed below.
        total_engine_init_s = 0.0
        total_fwd_time_s = 0.0
        total_pred_count = 0
        n_engine_builds = 0
        _timing_warmup_done = False

        # Bounded cache for the most "interesting" samples — ranked by
        # peak-normalised MAE so big-field plans don't dominate purely
        # because they deposit more dose. The probe sample is held
        # separately so it's always plotted regardless of its score.
        per_sample_metrics = []  # list of dicts with mae / lv1_beam_mae / lv1_idd / mask_*
        worst_cache = _TopKByMetric(
            max(0, args.num_logged_beams - (1 if probe_index is not None else 0))
        )
        probe_payload = None

        # Per-(patient, beam) plan accumulator. Each (pid, beam) is treated
        # as an independent plan: the three beams of a patient target
        # different lesions with different isocenters, so summing them
        # would superimpose three independent treatments and produce a
        # nonsense plan. Keys are (patient_id, beam_index).
        # (patient, modality) -> arrays summed over control points AND BEAMS.
        # The challenge's Level 2 is "the complete reconstructed treatment plan
        # ... combining beams with their clinical weights" -- ONE plan per
        # patient, not one per beam. Keying this by beam made every reported
        # plan metric a per-beam quantity: the strata are percentages of the
        # PRESCRIPTION, and a single beam never reaches 80% of it, so the high
        # stratum was near-empty and plan_mae_combined was computed over a
        # different voxel population than the metric it was named after. MU
        # weighting is already in the per-CP dose, so summing all CPs of all
        # beams is exactly the weighted plan.
        plan_accum = {}

        with torch.no_grad():
            # When --skip_validation, run zero val iterations; everything
            # downstream of this loop handles empty per_sample_metrics /
            # plan_accum cleanly (np.mean of empty -> NaN log, no plots).
            val_iter = iter([]) if args.skip_validation else iter(val_loader)
            for batch in val_iter:
                sample = batch[0]
                is_batched = sample.get("batched", False)
                if is_batched:
                    sample["beam"] = PDRT.BeamSequence.from_beams(sample["beam_list"])
                beam = sample["beam"].to(device)
                image = sample["image"].to(device)
                density_image = sample["density_image"].to(device)
                target_dose = sample["dose"].to(device)
                resolution = sample["resolution"]
                pad_info = sample["pad_info"]
                iso_center = sample["iso_center"]
                mask = sample["mask"].to(device)

                # Start the prediction timer BEFORE engine construction —
                # the kernel allocation / dose-correction wiring in
                # CorrectedDoseEngine.__init__ + .to(device) is part of
                # per-beam inference cost at submission time. Sync first
                # so this batch's start isn't billed any queued CUDA work
                # from the previous batch's evaluation.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                engine_start = time.time()

                engine = CorrectedDoseEngine(
                    machine_config=machine_config,
                    kernel_size=kernel_size,
                    dose_grid_shape=target_dose.shape[-3:],
                    dose_grid_spacing=resolution,
                    beam_template=beam,
                    fluence_correction_model=fluence_correction_model,
                    lateral_scatter_model=lateral_heterogeneity_model,
                    dose_correction_model=dose_correction_model,
                    fluence_to_dose=args.fluence_to_dose,
                    multislab=args.multislab,
                    mu_eff=args.mu_eff,
                    lattice_size=args.lattice_size,
                    lattice_depth_mode=args.lattice_depth_mode,
                    lattice_tile_chunk=args.lattice_tile_chunk,
                    terma_scaling=bool(args.terma),
                    terma_c1=(args.terma_c1 if args.terma else None),
                    terma_c2_per_mm=(args.terma_c2 if args.terma else None),
                    lateral_scatter=args.lateral_scatter,
                    v2_features=bool(args.v2_features),
                    v2_feature_cfg=v2_feature_cfg,
                    bev_crop=args.bev_crop,
                    bev_crop_margin_mm=args.bev_crop_margin_mm,
                    bev_crop_per_sample=args.bev_crop_per_sample,
                    bev_crop_min=args.bev_crop_min,
                    early_bev_crop=args.early_bev_crop,
                    amp_dtype=val_amp_dtype,   # fp16: matches the container
                    lat_sigma_mm=args.lat_sigma_mm,
                    lat_cap_mm=args.lat_cap_mm,
                    use_source_distance=args.use_source_distance,
                    device=device
                ).to(device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_engine_time = time.time() - engine_start
                fwd_start = time.time()

                if is_batched:
                    leaf_pos_in = beam.leaf_positions.unsqueeze(0)      # [1, G, N, 2]
                    mus_in      = beam.mus.unsqueeze(0)                 # [1, G]
                    jaws_in     = beam.jaw_positions.unsqueeze(0)       # [1, G, 2]
                else:
                    leaf_pos_in = beam.leaf_positions.unsqueeze(0).unsqueeze(0)
                    mus_in      = beam.mu.unsqueeze(0).unsqueeze(0)
                    jaws_in     = beam.jaw_positions.unsqueeze(0).unsqueeze(0)

                # Always return_per_beam=True so we get [1, G, D, H, W]
                # and can unpack the G dim below — works uniformly for
                # G=1 (single-CP) and G>1 (batched-CP) validation.
                pred_dose_all = engine.forward(
                    leaf_positions=leaf_pos_in,
                    mus=mus_in,
                    jaw_positions=jaws_in,
                    density_image=density_image.unsqueeze(0),
                    return_per_beam=True,
                    **({} if args.legacy_val_inputs else
                       _corrector_forward_kwargs(args, image, beam, is_batched)),
                )[0]                                                    # [G, D, H, W]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_fwd_time = time.time() - fwd_start

                # Normalise targets to [G, D, H, W].
                if target_dose.dim() == 3:
                    target_dose_all = target_dose.unsqueeze(0)
                else:
                    target_dose_all = target_dose
                G_local = pred_dose_all.shape[0]
                # Drop the first timed batch: cuDNN autotune, CUDA context and
                # first-touch allocation land there and are paid once per
                # process, not once per dose map.
                if _timing_warmup_done:
                    total_engine_init_s += batch_engine_time
                    total_fwd_time_s += batch_fwd_time
                    total_pred_count += G_local
                    n_engine_builds += 1
                else:
                    _timing_warmup_done = True

                # Mask + crop_and_pad inputs (per-CP slices computed in
                # the inner loop below). The G dim broadcasts against
                # the [D, H, W] mask.
                masked_pred_padded_all = pred_dose_all * mask.unsqueeze(0)   # [G, D, H, W]
                target_padded_all      = target_dose_all * mask.unsqueeze(0) # [G, D, H, W]

                # CT / body-mask diagnostics are shared across all G CPs
                # of this same-beam batch — compute once.
                image_fill = -1000.0   # HU air: the MR arm carries a synthetic CT here too
                ct_volume = crop_and_pad_to_original(image.detach().cpu().numpy(), pad_info, fill_value=image_fill)
                mask_orig = crop_and_pad_to_original(mask, pad_info, fill_value=False)
                mask_np_for_plot = mask_orig.detach().cpu().numpy()
                shared_mask_fraction = float(mask.float().mean().item())
                h_orig = pad_info["original_shape"][0]
                h_kept = h_orig + pad_info["pad_H_before"] + pad_info["pad_H_after"]
                shared_h_crop_frac = float(h_kept) / float(max(h_orig, 1))

                # BEV doses for IDD — all G inverse-rotated in one call.
                pred_bev_all   = engine.patient_to_bev(masked_pred_padded_all)   # [1, G, D, H, W]
                target_bev_all = engine.patient_to_bev(target_padded_all)       # [1, G, D, H, W]

                # Model correction in patient frame for plot panels
                # (per-CP — do NOT sum across G).
                if engine.dose_correction is not None:
                    # The corrector runs in BEV, so its output must be rotated
                    # into the patient frame before it is scored.
                    corr_patient_all = engine.rotation_layer(engine.dose_correction)
                else:
                    corr_patient_all = None

                # Fluence correction stat (per-batch, recorded G times so
                # the per-anatomy aggregator sees the same count of CPs).
                f_corr_frac = None
                if engine.fluence_map_correction is not None:
                    f_corr_frac = (
                        torch.sum(torch.abs(engine.fluence_map_correction)).item()
                        / torch.sum(torch.abs(engine.original_fluence_maps)).item()
                    )

                for g in range(G_local):
                    single_pred_orig   = crop_and_pad_to_original(masked_pred_padded_all[g], pad_info, fill_value=0.0)
                    single_target_orig = crop_and_pad_to_original(target_padded_all[g], pad_info, fill_value=0.0)

                    if f_corr_frac is not None:
                        mean_absolute_correction.append(f_corr_frac)
                    if engine.dose_correction is not None:
                        cp_correction = engine.dose_correction[0, g]
                        denom = max(torch.abs(single_pred_orig).sum().item(), 1e-12)
                        mean_absolute_dose_correction.append(
                            torch.abs(cp_correction).sum().item() / denom
                        )

                    metrics = _compute_metrics(single_pred_orig, single_target_orig)
                    loss = metrics["mae"]
                    metrics["lv1_beam_mae"] = _beam_masked_mae_normalized(
                        single_pred_orig, single_target_orig
                    )
                    # Patient space: this is the scored quantity. The BEV
                    # variant is kept alongside because it profiles along beam
                    # depth and so is the one that sees range/build-up error
                    # directly -- useful for diagnosis, not for scoring.
                    metrics["lv1_idd"] = _idd_distance(
                        single_pred_orig.float(), single_target_orig.float()
                    )
                    metrics["lv1_idd_bev"] = _idd_distance(
                        pred_bev_all[0, g].float(), target_bev_all[0, g].float()
                    )
                    metrics["leak_mass"], metrics["leak_idd"] = _zero_region_leak(
                        pred_bev_all[0, g].float(), target_bev_all[0, g].float()
                    )
                    metrics["mask_fraction"] = shared_mask_fraction
                    metrics["h_crop_frac"] = shared_h_crop_frac

                    # Plan accumulator: this CP's pred/true added to the running
                    # sum for its (patient, beam, modality). A "plan" here is ONE
                    # BEAM summed over all its control points — beams are NOT
                    # summed together (they have different isocentres). Gamma runs
                    # only for --gamma_beam (default 0), i.e. n_patients x
                    # n_modalities = 12 full-grid evaluations per epoch, and never
                    # on an individual control point.
                    idx_key = val_loader.dataset.index_list[num_batches]
                    pid_now, beam_idx_now = idx_key[0], idx_key[1]
                    mod_now = idx_key[3] if len(idx_key) > 3 else "ct"
                    plan_key = (pid_now, mod_now)
                    pred_np_now = single_pred_orig.detach().cpu().numpy()
                    true_np_now = single_target_orig.detach().cpu().numpy()
                    if plan_key not in plan_accum:
                        plan_accum[plan_key] = {
                            "pred": pred_np_now.astype(np.float32, copy=True),
                            "true": true_np_now.astype(np.float32, copy=True),
                            "ct": ct_volume.copy(),
                            "mask": mask_np_for_plot.copy(),
                            "iso_center": iso_center,
                            "anatomy": pid_now[:4],
                            "modality": mod_now,
                            "beam_index": beam_idx_now,
                        }
                    else:
                        plan_accum[plan_key]["pred"] += pred_np_now
                        plan_accum[plan_key]["true"] += true_np_now

                    # Plot payload. Building it is deferred to here-or-later
                    # via the lazy closure: make_payload() is only invoked
                    # for the probe CP and for CPs that win a slot in
                    # worst_cache (~K per epoch). All other CPs skip the
                    # correction-volume crop + host copy entirely. pred_np /
                    # true_np are already on the host from plan_accum above,
                    # so reuse them instead of re-copying inside the closure.
                    def make_payload(pred_np=pred_np_now, target_np=true_np_now,
                                     ct_v=ct_volume, iso=iso_center,
                                     corr_src=corr_patient_all, g_idx=g,
                                     msk=mask, pi=pad_info,
                                     mask_np=mask_np_for_plot,
                                     key=idx_key, m=metrics):
                        if corr_src is not None:
                            corr_t = crop_and_pad_to_original(
                                corr_src[0, g_idx] * msk, pi, fill_value=0.0
                            )
                            corr_np = corr_t.detach().cpu().numpy()
                        else:
                            corr_np = None
                        payload = _build_payload(
                            ct_v,
                            target_np,
                            pred_np,
                            iso,
                            corr_np,
                            mask_np,
                            key,
                        )
                        payload["metrics"] = m
                        return payload

                    if num_batches == probe_index:
                        probe_payload = make_payload()
                    else:
                        worst_cache.offer(metrics["lv1_beam_mae"], num_batches, make_payload)

                    per_sample_metrics.append(metrics)
                    val_loss_total += loss
                    num_batches += 1

                # Drop refs + clear cache once per batch (not per CP);
                # the large intermediates live on the engine, not on
                # per-CP slices.
                del engine, beam, image, density_image, target_dose, mask
                del pred_dose_all, masked_pred_padded_all, target_padded_all
                del pred_bev_all, target_bev_all
                if corr_patient_all is not None:
                    del corr_patient_all
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss = val_loss_total / max(num_batches, 1)

        # Anatomy-stratified validation metrics. Only the challenge scores
        # plus a couple of diagnostics — no nmae / hd_mae any more.
        score_keys = ("mae", "mask_fraction", "h_crop_frac",
                      "lv1_beam_mae", "lv1_idd", "leak_mass", "leak_idd")
        per_anat = {}
        per_mod = {}
        per_mod_anat = {}
        for idx, m in enumerate(per_sample_metrics):
            entry = val_loader.dataset.index_list[idx]
            prefix = entry[0][:4]
            mod_i = entry[3] if len(entry) > 3 else "ct"
            per_mod.setdefault(mod_i, {k: [] for k in score_keys})
            for k in score_keys:
                if k in m and m[k] == m[k]:
                    per_mod[mod_i][k].append(m[k])
            per_anat.setdefault(prefix, {k: [] for k in score_keys})
            for k in score_keys:
                per_anat[prefix][k].append(m[k])
            # Modality x anatomy. The two one-way splits above cannot show
            # whether an MR regression is thorax or abdomen, which is the
            # question the sCT arm actually raises.
            per_mod_anat.setdefault((mod_i, prefix), {k: [] for k in score_keys})
            for k in score_keys:
                if k in m and m[k] == m[k]:
                    per_mod_anat[(mod_i, prefix)][k].append(m[k])

        log_metrics = {
            "validation_loss": val_loss,
            "mean_absolute_correction": np.mean(mean_absolute_correction) if mean_absolute_correction else 0.0,
            "mean_absolute_dose_correction": np.mean(mean_absolute_dose_correction) if mean_absolute_dose_correction else 0.0,
            # Challenge's efficiency metric: average wall-clock time per
            # beam. Total prediction time (engine init + forward, synced
            # at both ends) summed across the whole val pass, divided by
            # the number of CPs predicted — so engine setup overhead is
            # billed correctly and the value reflects real per-beam cost.
            # Same NAME as before so the Comet history stays comparable, but
            # composed the way the leaderboard bills it:
            #     per dose map = forward per CP + (one engine build / num_cps)
            # rather than (build + forward) / batched_cp_size.
            "runtime_per_beam": (
                (total_fwd_time_s / total_pred_count)
                + (total_engine_init_s / n_engine_builds / max(1, args.num_cps))
            ) if total_pred_count and n_engine_builds else 0.0,
            # Components, for when a runtime difference needs explaining.
            "runtime_forward_per_cp": (total_fwd_time_s / total_pred_count) if total_pred_count else 0.0,
            "runtime_engine_build_s": (total_engine_init_s / n_engine_builds) if n_engine_builds else 0.0,
        }
        # Per-modality Level-1 means (ct vs mr) — the headline split.
        for mod_i, d in per_mod.items():
            for k, vals in d.items():
                if vals:
                    log_metrics[f"val_{k}_{mod_i}"] = float(np.mean(vals))
        # Per-modality x anatomy means.
        for (mod_i, prefix), d in per_mod_anat.items():
            for k, vals in d.items():
                if vals:
                    log_metrics[f"val_{k}_{mod_i}_{prefix}"] = float(np.mean(vals))
        # Overall means / std / extrema.
        for k in score_keys:
            vals = [m[k] for m in per_sample_metrics]
            if vals:
                log_metrics[f"val_{k}"] = float(np.mean(vals))
                log_metrics[f"val_{k}_std"] = float(np.std(vals))
                log_metrics[f"val_{k}_max"] = float(np.max(vals))
                if k == "mask_fraction":
                    log_metrics[f"val_{k}_min"] = float(np.min(vals))
        # Per-anatomy means / std.
        for prefix, by_key in per_anat.items():
            for k, vals in by_key.items():
                log_metrics[f"val_{k}_{prefix}"] = float(np.mean(vals))
                log_metrics[f"val_{k}_std_{prefix}"] = float(np.std(vals))
                log_metrics[f"val_{k}_max_{prefix}"] = float(np.max(vals))
                if k == "mask_fraction":
                    log_metrics[f"val_{k}_min_{prefix}"] = float(np.min(vals))
        experiment.log_metrics(log_metrics, epoch=epoch)

        # Overall lvl1 beam MAE across all validation samples (the challenge lvl1 metric);
        # used to select the best checkpoint.
        _lv1_vals = [m["lv1_beam_mae"] for m in per_sample_metrics
                     if isinstance(m.get("lv1_beam_mae"), float) and m["lv1_beam_mae"] == m["lv1_beam_mae"]]
        cur_lv1_mae = float(np.mean(_lv1_vals)) if _lv1_vals else float("nan")

        print(f"Validation Loss: {val_loss:.4f}")

        def _metric_row(label, by_key):
            if not by_key.get("mae"):
                return
            print(
                f"  {label}: MAE={np.mean(by_key['mae']):.4f}±{np.std(by_key['mae']):.4f}  "
                f"lv1_beam_mae={np.mean(by_key['lv1_beam_mae']):.4f}±{np.std(by_key['lv1_beam_mae']):.4f}  "
                f"lv1_idd={np.mean(by_key['lv1_idd']):.4f}±{np.std(by_key['lv1_idd']):.4f}  "
                f"leak_mass={np.mean(by_key['leak_mass']):.4f}  "
                f"leak_idd={np.mean(by_key['leak_idd']):.4f}  "
                f"mask={np.mean(by_key['mask_fraction']):.3f}"
                f" (min={np.min(by_key['mask_fraction']):.3f},"
                f" max={np.max(by_key['mask_fraction']):.3f})  "
                f"n={len(by_key['mae'])}"
            )

        # Stratified BOTH ways: each arm gets its anatomies and then its own
        # mean, so an MR-only regression is attributable to a site rather than
        # averaged away against CT.
        for mod_i in sorted(per_mod):
            for prefix in sorted(p_ for (m_, p_) in per_mod_anat if m_ == mod_i):
                _metric_row(f"{mod_i.upper()} {prefix}", per_mod_anat[(mod_i, prefix)])
            _metric_row(f"{mod_i.upper()} mean", per_mod[mod_i])
        if len(per_mod) > 1:
            for prefix, by_key in sorted(per_anat.items()):
                _metric_row(f"all {prefix}", by_key)

        # Flag samples with implausible mask coverage so broken body masks
        # don't hide in the aggregate. Empirically a sensible body mask is
        # somewhere between 5% and 95% of the padded volume; outside that
        # range either the threshold is wrong or the largest-component step
        # picked up the wrong blob.
        suspicious = [
            (val_loader.dataset.index_list[i], m["mask_fraction"])
            for i, m in enumerate(per_sample_metrics)
            if m["mask_fraction"] < 0.05 or m["mask_fraction"] > 0.95
        ]
        if suspicious:
            print(f"  body_mask outliers ({len(suspicious)}):")
            for _entry, frac in suspicious[:10]:
                patient, b, cp = _entry[0], _entry[1], _entry[2]
                print(f"    {patient} B{b} CP{cp:03d}  mask_fraction={frac:.3f}")
            try:
                experiment.log_text(
                    "\n".join(
                        f"{p} B{b} CP{cp:03d}  {f:.3f}" for (p, b, cp), f in suspicious
                    ),
                    step=epoch,
                    metadata={"kind": "mask_outliers", "epoch": epoch},
                )
            except Exception:
                pass

        # Per-sample histograms so you can spot tails on each metric.
        try:
            for k in score_keys:
                experiment.log_histogram_3d(
                    [m[k] for m in per_sample_metrics], name=f"val_{k}_hist", step=epoch,
                )
        except Exception:
            pass

        # Probe first, then the worst-by-nMAE samples (bounded by the heap).
        to_log = []
        if probe_payload is not None:
            to_log.append(("probe", probe_index, probe_payload))
        for idx, payload in worst_cache.items():
            to_log.append(("worst_lv1_beam_mae", idx, payload))

        for kind, idx, sample in to_log:
            m = sample["metrics"]
            tag = (
                f"{sample['patient_id']}_B{sample['beam_index']}"
                f"_CP{sample['cp_index']:03d}"
            )
            out_path = f"out/val_{tag}.png"
            plot_dose_comparison(
                sample["ct_volume"],
                sample["true_dose"],
                sample["pred_dose"],
                None,
                sample["iso_center"],
                save_path=out_path,
                title=(
                    f"{tag}   MAE={m['mae']:.4f}  "
                    f"lv1_beam_mae={m['lv1_beam_mae']:.4f}  "
                    f"lv1_idd={m['lv1_idd']:.4f}  "
                    f"mask={m.get('mask_fraction', float('nan')):.3f}"
                ),
                correction=sample["correction"],
                body_mask=sample.get("body_mask"),
            )
            experiment.log_image(out_path, name=f"{kind}_{tag}.png", step=epoch)

        # Free cached arrays.
        probe_payload = None
        worst_cache = None

        # --- Level-2 challenge metrics: per-plan evaluation -------------------
        # plan_accum is keyed by (patient, beam, modality); each entry is one
        # independent plan. We compute stratified MAE and gamma for every
        # plan, log every plan's numbers, and plot only the K most
        # informative plans (probe + worst by combined MAE) to keep the
        # Comet image count manageable.
        plan_metrics = {}  # (patient, beam, modality) -> dict
        # Gamma is the expensive part of validation (~70 min for the 12 plans),
        # so it runs on a cadence. The final epoch always runs it so the last
        # reported number is complete. Skipped epochs log NaN, which the
        # aggregation and the best-checkpoint logic already handle.
        _n = int(args.gamma_every_n_epochs)
        gamma_due = (_n == 1) or (_n > 0 and (epoch % _n == 0 or epoch == num_epochs - 1))
        if not args.no_gamma and not gamma_due:
            nxt = (f"epoch {((epoch // _n) + 1) * _n}" if _n > 0 else "never (0 = disabled)")
            print(f"Plan-level: gamma skipped this epoch "
                  f"(--gamma_every_n_epochs {_n}); next at {nxt}.")
        for plan_key, plan in plan_accum.items():
            pid, mod = plan_key
            pred_plan = plan["pred"]
            true_plan = plan["true"]

            # Passive calibration sanity check: the least-squares scale
            # that would best match pred to true. With the engine now
            # calibrated this should sit near 1.0; persistent drift means
            # the baked-in calibration no longer fits and should be
            # revisited. NOT applied — Level-2 metrics are absolute.
            scale_plan = _ls_scale(pred_plan, true_plan)

            # Prescription proxy: peak of the GT plan dose. This stands in for
            # the clinical prescription (which we don't have without segs).
            rx = float(true_plan.max())
            strat = _stratified_plan_mae(pred_plan, true_plan, rx)
            # Gamma only for --gamma_beam: n_patients x n_modalities = 12
            # evaluations per validation cycle, on the FULL summed-control-point
            # beam dose vs the reference, never per control point and never
            # subsampled. Settings match evaluation/.../metrics_plan.py exactly.
            # Measured cost: ~350 s per call on an 11 M-voxel plan, i.e. ~70 min
            # per validation. That is inherent to 1%/1mm with interp_fraction=10
            # on a 2 mm grid — cropping to the evaluated region (10x fewer
            # voxels) and ram_available both make no difference, and lowering
            # interp_fraction changes the answer (it has not converged at 10).
            # The
            # returned gamma_map is dense enough to render in the plan
            # plot — and reuse it directly downstream instead of running
            # a SECOND (full-grid, multi-minute) gamma pass per plotted
            # plan, which was starving the GPU during validation.
            # --gamma_beam is inert now: a plan is all beams summed, so there
            # is exactly one plan per (patient, modality) and nothing to filter.
            if args.no_gamma or not gamma_due:
                gamma = float("nan")
                gamma_map = None
            else:
                try:
                    gamma, gamma_map = _gamma_pass_rate_3d(
                        pred_plan, true_plan,
                        voxel_mm=2.0,
                        prescription=rx,
                        random_subset=None,     # full volumes, always
                        return_map=True,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  gamma failed for {pid} {mod}: {exc!r}")
                    traceback.print_exc()
                    gamma = float("nan")
                    gamma_map = None

            plan_metrics[plan_key] = {
                "anatomy": plan["anatomy"],
                "plan_mae_high": strat["high"],
                "plan_mae_mid": strat["mid"],
                "plan_mae_low": strat["low"],
                "plan_mae_combined": strat["combined"],
                "plan_gamma_1_1": gamma,
                "gamma_map": gamma_map,
                "scale_factor_plan": scale_plan,
                "prescription": rx,
            }

        # Decide which plans to actually render. The probe beam's plan is
        # always plotted (consistent reference across epochs), then we fill
        # up to num_logged_beams with the worst-combined-MAE plans.
        # Plot every plan. Probe goes first so it's the top entry in Comet's
        # image list each epoch; the rest follow in descending combined-MAE
        # order so the "interesting" ones are still near the top.
        probe_plan_key = None
        if probe_index is not None:
            pkey = val_loader.dataset.index_list[probe_index]
            candidate = (pkey[0], pkey[1], pkey[3] if len(pkey) > 3 else "ct")
            if candidate in plan_metrics:
                probe_plan_key = candidate

        ordered = sorted(
            plan_metrics.items(),
            key=lambda kv: -(kv[1]["plan_mae_combined"]),
        )
        plans_to_plot = []
        if probe_plan_key is not None:
            plans_to_plot.append(probe_plan_key)
        for k, _m in ordered:
            if k == probe_plan_key:
                continue
            plans_to_plot.append(k)

        print(f"Plan-level: accumulated {len(plan_accum)} plans, "
              f"metrics for {len(plan_metrics)}, plotting {len(plans_to_plot)}.")

        plot_ok = 0
        for plan_key in plans_to_plot:
            pid, mod = plan_key
            m = plan_metrics[plan_key]
            plan = plan_accum[plan_key]
            tag = f"{pid}_{mod.upper()}"
            # Gamma map already cached on plan_metrics by the metric pass
            # reuse it instead of doing a second
            # full-grid pymedphys.gamma per plot, which was starving the
            # GPU during validation.
            gamma_map = m.get("gamma_map")
            try:
                out_path = f"out/val_plan_{tag}.png"
                plot_dose_comparison(
                    plan["ct"],
                    plan["true"],
                    plan["pred"],
                    None,
                    plan["iso_center"],
                    save_path=out_path,
                    title=(
                        f"{tag} PLAN   combined_MAE={m['plan_mae_combined']:.4f}  "
                        f"γ(1%/1mm)={m['plan_gamma_1_1']:.2f}%   "
                        f"scale={m['scale_factor_plan']:.3f}   "
                        f"Rx≈{m['prescription']:.3f}"
                    ),
                    correction=None,
                    body_mask=plan["mask"],
                    gamma_map=gamma_map,
                )
                kind = "plan_probe" if plan_key == probe_plan_key else "plan_worst"
                experiment.log_image(out_path, name=f"{kind}_{tag}.png", step=epoch)
                plot_ok += 1
            except Exception as exc:
                import traceback
                print(f"  plan plot failed for {tag}: {exc!r}")
                traceback.print_exc()
        print(f"Plan-level: logged {plot_ok}/{len(plans_to_plot)} plots to Comet.")
        # Drop the cached gamma_map arrays now that plots are done — they
        # can be hundreds of MB across all plans, and only the scalar
        # metric is needed downstream.
        for m in plan_metrics.values():
            m.pop("gamma_map", None)

        # Aggregate Level-2 metrics: average across plans, plus per anatomy.
        cur_gamma = None  # populated below if plan_metrics is non-empty
        if plan_metrics:
            # scale_factor_plan is deliberately NOT here. It fanned out to ~8
            # Comet metrics an epoch (overall, std, per-modality, per-anatomy)
            # for a diagnostic that only ever mattered during calibration, and
            # calibration is closed. Still computed and shown on the plan plots.
            lv2_keys = ("plan_mae_high", "plan_mae_mid", "plan_mae_low",
                        "plan_mae_combined", "plan_gamma_1_1")
            lv2_log = {}
            for k in lv2_keys:
                vals = [m[k] for m in plan_metrics.values()
                        if not (isinstance(m[k], float) and (m[k] != m[k]))]
                if vals:
                    lv2_log[f"lv2_{k}"] = float(np.mean(vals))
                    lv2_log[f"lv2_{k}_std"] = float(np.std(vals))
            # Per-modality (ct vs mr): the headline split now that one
            # corrector serves both arms.
            by_mod_lv2 = {}
            for m in plan_metrics.values():
                by_mod_lv2.setdefault(m.get("modality", "ct"), []).append(m)
            for mod_name, ms in by_mod_lv2.items():
                for k in lv2_keys:
                    vals = [x[k] for x in ms if not (isinstance(x[k], float) and x[k] != x[k])]
                    if vals:
                        lv2_log[f"lv2_{k}_{mod_name}"] = float(np.mean(vals))
            # Per-anatomy
            by_anat_lv2 = {}
            for m in plan_metrics.values():
                by_anat_lv2.setdefault(m["anatomy"], []).append(m)
            for anat, ms in by_anat_lv2.items():
                for k in lv2_keys:
                    vals = [m[k] for m in ms
                            if not (isinstance(m[k], float) and (m[k] != m[k]))]
                    if vals:
                        lv2_log[f"lv2_{k}_{anat}"] = float(np.mean(vals))
                        lv2_log[f"lv2_{k}_std_{anat}"] = float(np.std(vals))
            cur_gamma = lv2_log.get("lv2_plan_gamma_1_1")
            experiment.log_metrics(lv2_log, epoch=epoch)

            print(f"Level-2 plan metrics (mean±std across {len(plan_metrics)} plans):")
            for k in lv2_keys:
                if f"lv2_{k}" in lv2_log:
                    print(f"  {k}: {lv2_log[f'lv2_{k}']:.4f}±{lv2_log.get(f'lv2_{k}_std', 0.0):.4f}")

            # Per-plan gamma and plan MAE. Only the mean and per-anatomy mean
            # were printed, and with a std of ~9 points across 6 plans the mean
            # hides whether a gap is systematic or one bad patient.
            print(f"  per-plan gamma / plan_mae_combined:")
            for (pid, mod), m in sorted(plan_metrics.items()):
                g = m.get("plan_gamma_1_1", float("nan"))
                c = m.get("plan_mae_combined", float("nan"))
                print(f"    {pid}_{mod.upper()}: "
                      f"gamma={g:.2f}%  plan_mae_combined={c:.5f}")

        # Release plan accumulator memory.
        plan_accum.clear()

        # Save ONLY the learnable correction weights — not the optimizer
        # state or any engine/physics tensors. The CorrectedDoseEngine is
        # rebuilt from configs.py at load time; all we need to resume or
        # serve is the correction net's weights plus a little provenance.
        ckpt = {
            "epoch": epoch + 1,
            "dose_model_state_dict": dose_correction_model.state_dict() if dose_correction_model is not None else None,
            "fluence_model_state_dict": fluence_correction_model.state_dict() if fluence_correction_model is not None else None,
            "lateral_heterogeneity_model_state_dict": (
                lateral_heterogeneity_model.state_dict()
                if lateral_heterogeneity_model is not None else None),
            "val_loss": val_loss,
            "experiment_name": run_name,
            "experiment_key": experiment.get_key(),
            "model_config": corrector_model_config,
        }
        torch.save(ckpt, ckpt_path)

        # Track best checkpoint by mean Level-2 gamma (1%/1mm) pass rate.
        # cur_gamma is set when plan_metrics is non-empty (i.e. validation
        # actually produced a finished plan); gamma can also be NaN if
        # pymedphys is missing or a plan failed. Skip the best-ckpt
        # update on either condition.
        if cur_lv1_mae == cur_lv1_mae and cur_lv1_mae < best_val_lv1:
            best_val_lv1 = cur_lv1_mae
            ckpt["best_val_lv1_beam_mae"] = best_val_lv1
            if cur_gamma is not None and cur_gamma == cur_gamma:
                ckpt["best_val_plan_gamma_1_1"] = cur_gamma
            torch.save(ckpt, ckpt_best_path)
            print(f"  new best val_lv1_beam_mae={best_val_lv1:.4f} at epoch {epoch}")

        # Restore the live (non-EMA) weights for training.
        if ema is not None:
            ema.swap_out(_ema_target)

        if dose_correction_model is not None:
            dose_correction_model.train()
        if fluence_correction_model is not None and not args.tiny_fluence_ckpt:
            fluence_correction_model.train()

        num_batches = 0
        optimizer.zero_grad()
        max_steps = args.max_train_steps
        # Per-section timing (only collected when --profile_steps > 0).
        profile_every = max(0, int(args.profile_steps))
        prof = {"data": 0.0, "engine_init": 0.0, "forward": 0.0,
                "backward": 0.0, "optim": 0.0, "n": 0}
        loader_iter = iter(loader)
        _t0 = time.time()
        for _ in range(10**9):
            if max_steps is not None and num_batches >= max_steps:
                break
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            if profile_every:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _t_data_end = time.time(); prof["data"] += _t_data_end - _t0

            sample = batch[0]
            is_batched = sample.get("batched", False)
            if is_batched:
                sample["beam"] = PDRT.BeamSequence.from_beams(sample["beam_list"])
            beam = sample["beam"].to(device)
            image = sample["image"].to(device)
            density_image = sample["density_image"].to(device)
            target_dose = sample["dose"].to(device)
            resolution = sample["resolution"]
            pad_info = sample["pad_info"]
            iso_center = sample["iso_center"]
            mask = sample["mask"].to(device)

            engine = CorrectedDoseEngine(
                machine_config=machine_config,
                kernel_size=kernel_size,
                dose_grid_shape=target_dose.shape[-3:],
                dose_grid_spacing=resolution,
                beam_template=beam,
                fluence_correction_model=fluence_correction_model,
                lateral_scatter_model=lateral_heterogeneity_model,
                dose_correction_model=dose_correction_model,
                fluence_to_dose=args.fluence_to_dose,
                multislab=args.multislab,
                mu_eff=args.mu_eff,
                lattice_size=args.lattice_size,
                lattice_depth_mode=args.lattice_depth_mode,
                lattice_tile_chunk=args.lattice_tile_chunk,
                terma_scaling=bool(args.terma),
                terma_c1=(args.terma_c1 if args.terma else None),
                terma_c2_per_mm=(args.terma_c2 if args.terma else None),
                lateral_scatter=args.lateral_scatter,
                v2_features=bool(args.v2_features),
                v2_feature_cfg=v2_feature_cfg,
                bev_crop=args.bev_crop,
                bev_crop_margin_mm=args.bev_crop_margin_mm,
                bev_crop_per_sample=args.bev_crop_per_sample,
                bev_crop_min=args.bev_crop_min,
                early_bev_crop=args.early_bev_crop,
                amp_dtype=amp_dtype,
                lat_sigma_mm=args.lat_sigma_mm,
                lat_cap_mm=args.lat_cap_mm,
                use_source_distance=args.use_source_distance,
                device=device
            ).to(device)
            if profile_every:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _t_init_end = time.time(); prof["engine_init"] += _t_init_end - _t_data_end

            if is_batched:
                leaf_pos_in = beam.leaf_positions.unsqueeze(0)        # [1, G, N, 2]
                mus_in      = beam.mus.unsqueeze(0)                   # [1, G]
                jaws_in     = beam.jaw_positions.unsqueeze(0)         # [1, G, 2]
            else:
                leaf_pos_in = beam.leaf_positions.unsqueeze(0).unsqueeze(0)
                mus_in      = beam.mu.unsqueeze(0).unsqueeze(0)
                jaws_in     = beam.jaw_positions.unsqueeze(0).unsqueeze(0)

            # Per-CP supervision when batched (returns [B, G, D, H, W]);
            # single-CP keeps the old summed [B, D, H, W] output.
            fwd_kw = _corrector_forward_kwargs(args, image, beam, is_batched)
            pred_dose = engine.forward(
                leaf_positions=leaf_pos_in,
                mus=mus_in,
                jaw_positions=jaws_in,
                density_image=density_image.unsqueeze(0),
                return_per_beam=is_batched,
                **fwd_kw,
            ) * mask                                                      # mask [D,H,W] broadcasts
            if profile_every:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _t_fwd_end = time.time(); prof["forward"] += _t_fwd_end - _t_init_end

            target_dose_b = target_dose.unsqueeze(0)
            if use_v2_loss:
                # Aux predictions from the coarse decoder levels, non-empty
                # only under --v1_deep_supervision and only in training mode.
                # They are CORRECTED DOSE in BEV at reduced resolution, which
                # is why the target is rotated to BEV below and pooled down to
                # meet them inside the loss.
                deep = getattr(engine, "dose_correction_deep", ())
                deep_target = None
                if deep:
                    # Deep-supervision outputs are in BEV, so the target has to be
                    # rotated to match rather than the predictions rotated back.
                    with torch.no_grad():
                        deep_target = engine.patient_to_bev(
                            target_dose_b if target_dose_b.dim() == 4
                            else target_dose_b.squeeze(0))
                        # Under --bev_crop the aux predictions live on the
                        # CROPPED grid. Pooling a full-size target onto them
                        # would land spatially offset -- the same class of
                        # error as forgetting to crop away the U-Net padding.
                        _cs = getattr(engine, "bev_crop_slices", None)
                        if _cs is not None:
                            deep_target = deep_target[..., _cs[0], _cs[1], _cs[2]]
                loss, loss_terms = photon_corrector_loss(
                    pred_dose, target_dose_b,
                    deep_predictions=deep, deep_target=deep_target,
                    cfg=v2_loss_cfg,
                )
                if args.v2_w_local > 0:
                    # Layered here rather than inside losses_v2 so that module
                    # stays exactly as the reference implementation shipped it.
                    _t = target_dose_b.reshape(target_dose_b.shape[0], -1) \
                        if target_dose_b.dim() > 3 else target_dose_b.reshape(1, -1)
                    _p = pred_dose.reshape(_t.shape)
                    _peak = _t.detach().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                    _floor = 0.1 * _peak
                    _m = (_t >= _floor).to(_t.dtype)
                    _rel = (_p - _t).abs() / _t.detach().clamp_min(_floor)
                    _lt = (_rel * _m).sum() / _m.sum().clamp_min(1.0)
                    loss = loss + args.v2_w_local * _lt
                    loss_terms["local"] = float(_lt.detach())
            else:
                loss = criterion(pred_dose, target_dose_b)
            if l1_norm_alpha and engine.dose_correction is not None:
                loss = loss + l1_norm_alpha * engine.dose_correction.abs().sum()

            (loss / accumulation_steps).backward()
            if profile_every:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _t_bwd_end = time.time(); prof["backward"] += _t_bwd_end - _t_fwd_end

            # Detach instead of .item() — .item() forces a host-side
            # sync every step. We materialise the floats once at the
            # end of the epoch.
            training_loss.append(loss.detach())
            num_batches += 1

            if num_batches % accumulation_steps == 0:
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [q for g in optimizer.param_groups for q in g["params"]],
                        args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if ema is not None:
                    ema.update(_ema_target)
                # Periodic allocator drain: every batch's H-crop is a
                # slightly different shape, so the CUDA allocator
                # gradually fragments. Flush every 100 optimizer
                # steps to bound that without per-iter sync cost.
                if torch.cuda.is_available() and (num_batches // accumulation_steps) % 100 == 0:
                    torch.cuda.empty_cache()
            if profile_every:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _t_opt_end = time.time(); prof["optim"] += _t_opt_end - _t_bwd_end
                prof["n"] += 1
                if prof["n"] >= profile_every:
                    tot = sum(v for k, v in prof.items() if k != "n")
                    msg = " | ".join(
                        f"{k}={prof[k]/prof['n']*1000:.1f}ms "
                        f"({prof[k]/tot*100:.0f}%)"
                        for k in ("data", "engine_init", "forward", "backward", "optim")
                    )
                    # Running peak VRAM. Not reset between prints: the number
                    # that decides whether an arm fits a card is the worst BEV
                    # box the run has met so far, and that arrives late --
                    # 556917 (base_channels 16) trained 29 min before it OOM'd.
                    # A per-window peak would have hidden exactly that.
                    if torch.cuda.is_available():
                        msg += (f" | peakGB={torch.cuda.max_memory_allocated()/2**30:.2f}"
                                f"/{torch.cuda.max_memory_reserved()/2**30:.2f}")
                    print(f"[profile step {num_batches}] {msg}")
                    prof = {k: 0.0 for k in prof}; prof["n"] = 0
                _t0 = time.time()

            # Release Python refs so the GPU allocator can recycle the
            # engine's intermediates between iters.
            del engine, beam, image, density_image, target_dose, mask, pred_dose, loss

        if num_batches % accumulation_steps != 0:
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [q for g in optimizer.param_groups for q in g["params"]],
                    args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if ema is not None:
                ema.update(_ema_target)

        # Materialise per-step GPU losses once (single sync) and average.
        if training_loss:
            if torch.is_tensor(training_loss[0]):
                mean_train_loss = float(torch.stack(training_loss).mean().item())
            else:
                mean_train_loss = float(np.mean(training_loss))
        else:
            mean_train_loss = float("nan")
        print(f"Training loss: {mean_train_loss:.4f}")
        experiment.log_metrics({"training_loss": mean_train_loss}, epoch=epoch)

        # ---- Post-training checkpoint -------------------------------------
        # The weights here have seen epoch+1 epochs of training. The validation
        # at the top of the NEXT epoch is what scores them; on the last epoch
        # there is no next iteration, so without this save those steps are
        # thrown away entirely.
        #
        # LIVE weights, not EMA: `swap_out` above restored them, and a resume
        # has to continue from the weights the optimiser's momentum belongs to.
        # The EMA shadow rides along separately so the average survives too.
        last_ckpt = {
            "epoch": epoch + 1,
            "epochs_trained": epoch + 1,
            "dose_model_state_dict": (dose_correction_model.state_dict()
                                      if dose_correction_model is not None else None),
            "fluence_model_state_dict": (fluence_correction_model.state_dict()
                                         if fluence_correction_model is not None else None),
            "lateral_heterogeneity_model_state_dict": (
                lateral_heterogeneity_model.state_dict()
                if lateral_heterogeneity_model is not None else None),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "train_loss": mean_train_loss,
            "best_val_lv1_beam_mae": best_val_lv1,
            "experiment_name": run_name,
            "experiment_key": experiment.get_key(),
            "model_config": corrector_model_config,
        }
        # Write-then-rename: a job killed mid-save would otherwise leave a
        # truncated file exactly where --resume looks for one.
        _tmp = ckpt_last_path + ".tmp"
        torch.save(last_ckpt, _tmp)
        os.replace(_tmp, ckpt_last_path)
        print(f"  saved {ckpt_last_path} after epoch {epoch} "
              f"(optimizer + scheduler"
              f"{' + ema' if ema is not None else ''} included)")
