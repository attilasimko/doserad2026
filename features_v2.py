"""BEV input-feature stack for the v2 photon corrector.

Kept separate from ``engines.py`` and free of any engine object so it can be
unit-tested on plain tensors. The engine's only job is to supply BEV volumes
plus the three axis coordinate vectors; every convention lives here.

BEV axis order is ``[B, C, D, H, W]``:

  * ``D`` -- along the beam. Attenuation, buildup, long-range scatter.
  * ``H`` -- patient z, the MLC LEAF axis (5 mm leaf pitch vs 2 mm voxels).
  * ``W`` -- in-plane lateral. Leaf ends and jaw edges, i.e. the sharp penumbra.

Position encoding is deliberately the SIMPLE Cartesian form (offsets from the
central axis plus a normalised depth index), mirroring the proton
``_build_bev_features``. Fan/divergent coordinates are held for a later
ablation; the divergence is used only where it is physically unavoidable, which
is projecting the aperture along diverging rays -- and that projection is done
by the engine's fluence-volume layer, not here.

FEATURE SETS
------------

``v2`` is the 12-channel stack the shipped ``photon_corrector.pt`` was trained
on. It MUST stay the default: it is what every existing checkpoint was fitted
to, and ``model_config`` on the older checkpoints carries no ``feature_set``
key at all.

``v3`` trades one channel for another, so the count is unchanged at 12:

  * DROPS ``depth_index``. It is ``linspace(0, 1)`` over the BEV crop, i.e. the
    only channel whose meaning depends on the crop size -- a different BEV box
    at inference makes it a different function of the same voxel. Physical
    depth is already carried by ``geometric_depth`` (mm from isocentre) and
    radially by ``source_distance``.
  * ADDS ``fluence`` -- the projected aperture, peak-normalised. The engine's
    fluence-volume layer has already cast the MLC opening along DIVERGING rays,
    which is the one place the beam's divergence genuinely matters and which no
    Cartesian coordinate channel can reproduce. It is a byproduct of the
    physics, so it costs nothing to compute.

Why the CONTINUOUS fluence and not the binary ``aperture_state`` the v2 trunk
consumed as an embedding: the deployed fractional-pixel-overlap aperture
already carries a sub-voxel soft edge, and thresholding at 0.5 throws exactly
that away. Level 1 is scored at 1 mm DTA on a 2 mm grid, so the sub-voxel edge
is the part that matters. Small apertures are both the worst MAE band (0.073 vs
0.040) and the worst scale band (ls_scale 0.921 vs 1.003), and area alone
cannot separate a 500 mm^2 slit from a 500 mm^2 square.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from materials_v2 import (
    material_id_from_hu,
    quantize_density,
    relative_electron_density,
)
# Aperture state ids emitted by ``build_bev_features``. A voxel's ray either
# passes through the open aperture or it does not.
#
# NOTHING CONSUMES THESE. They were the deferred v2 trunk's aperture embedding
# input; the v1 trunk has no embedding path for them, and ``feature_set="v3"``
# supplies the aperture as the CONTINUOUS ``fluence`` channel instead, which is
# strictly more informative at the same cost. Kept only so the v2-trunk
# ablation stays expressible from this builder.
APERTURE_OUTSIDE = 0
APERTURE_OPEN = 1
N_APERTURE_STATES_SIMPLE = 2

# Length scale used to normalise every distance channel, in mm. 100 mm keeps
# lateral offsets, depths and path lengths all O(1) for a typical crop, same
# role as the proton feature builder's depth_scale.
LENGTH_SCALE_MM = 100.0


#: Scalar-channel count per feature set, for cross-checking a checkpoint's
#: ``model_config`` against the builder that will feed it. Both sets emit 12,
#: so the count alone CANNOT distinguish them -- exactly the silent-mismatch
#: shape that let a corrector be served on the wrong baseline. The name
#: has to be carried in the checkpoint and compared explicitly.
BEV_FEATURE_SETS = ("v2", "v3", "v4")
COND_REGIONS = ("dose", "aperture")


@dataclass
class FeatureConfig:
    """Which channels to emit. Each flag is an ablation axis (stage D)."""

    # "v2" = the shipped 12-channel stack. "v3" = drop depth_index, add the
    # projected aperture (fluence). See the module docstring. Set in
    # __post_init__, which is authoritative: use_depth_index and use_fluence
    # are DERIVED, so they cannot drift out of sync with the set name that
    # gets written into the checkpoint.
    #
    # "v4" = v3 minus the six channels that were MEASURED not to earn their
    # place (analysis/ablate_corrector_inputs.py, scored against the reference
    # dose on 24 CPs across 6 patients; baseline masked MAE 0.9947% of peak):
    #
    #     lat_h            +0.0015    cond_log_area    +0.0075
    #     cond_mean_rho    +0.0050    lat_w            +0.0112
    #     source_distance  +0.0127    cond_lung_frac   +0.0339
    #
    # All six removed together cost +0.0443%, about 4% of the model's own
    # error. `source_distance` is pure redundancy: it correlates r = 1.000 with
    # geometric_depth, and the pair costs LESS to remove (+0.51) than
    # geometric_depth alone (+0.54). The three cond_* scalars sit at 1.19-1.23x
    # fresh initialisation -- they never moved off random.
    #
    # v4 is a NEW NAME rather than a redefinition of v3 on purpose: the set
    # name is what every checkpoint's model_config records, so redefining one
    # in place silently changes what already-trained weights were fitted to.
    feature_set: str = "v2"

    use_density: bool = True
    use_electron_density: bool = True
    use_cumulative_path: bool = True
    use_geometric_depth: bool = True
    use_position: bool = True          # lat_h, lat_w
    use_depth_index: bool = True       # derived from feature_set
    use_fluence: bool = False          # derived from feature_set
    use_source_distance: bool = True
    use_aperture: bool = True
    use_material: bool = True
    # Measure lat_h / lat_w from the CENTRE OF THE SUPPLIED EXTENT rather than
    # from the central axis.
    #
    # Only meaningful with --bev_crop, whose box is centred on the field. With
    # absolute offsets a field that sits off-axis presents the SAME local
    # physics at a different channel value, so the network has to relearn it at
    # every lateral position. Centring makes the crop translation-invariant,
    # which is the one symmetry a convnet gets for free.
    #
    # Nothing is added to carry the absolute position, because nothing needs
    # to: `source_distance` is built from the UN-offset coordinates, so it
    # still encodes how far off-axis a voxel is. It is also implicit in the v3
    # fluence channel, whose projected aperture drifts laterally with depth for
    # an off-axis field. So this is a pure re-centring -- the channel count is
    # unchanged and no information is lost.
    crop_relative_position: bool = False
    quantize_density_to_sim: bool = True
    aperture_threshold: float = 0.5
    conditioning: bool = True
    # The conditioning scalars are averaged over THE REGION THE FIELD SEES:
    # the projected open aperture intersected with the patient. Air is
    # rho_e ~ 0 and so passes any "is this lung?" test; cond_air_rho is the
    # floor that keeps it out. cond_lung_rho is the lung/tissue split.
    cond_air_rho: float = 0.05
    cond_lung_rho: float = 0.6
    # WHICH region. "dose" is the original definition and stays the default so
    # that every checkpoint trained before this existed keeps evaluating on the
    # inputs it was trained on. "aperture" is the corrected one. This is a
    # SILENT TRAINING CONTRACT -- it changes what the model was fitted to and
    # leaves no trace in the weights, exactly like bounded_residual -- so it is
    # recorded in the checkpoint's model_config and must be passed explicitly
    # to serve or evaluate a model trained with it.
    cond_region: str = "dose"
    # HOW the 3 conditioning scalars reach the network.
    #
    #   "film"     FiLM layers inside every residual block (the original).
    #   "channels" broadcast as 3 extra INPUT channels, no FiLM anywhere.
    #   "off"      not supplied at all.
    #
    # "channels" is the proton remedy. FiLM applies an unbounded per-channel
    # scale and shift, `x * (1 + scale) + shift`, once per block -- 14 times in
    # this trunk -- all driven by a 3-vector that is a GLOBAL property of the
    # control point. Nothing constrains it after the zero-init, so it is a
    # direct route to whole-volume coherent error: exactly what IDD punishes
    # and a per-voxel mean cannot see. As input channels the same information
    # is available but has to compete with local evidence at every conv.
    # "channels" is the default because it is what the surviving trunk can
    # actually use. "film" emitted a separate cond vector for the deferred v2
    # trunk's FiLM layers; on the v1 trunk it silently DROPS the 3 scalars
    # (10 input channels instead of 13). It is retained here only so the
    # ablation that measured FiLM remains expressible, and is deliberately not
    # offered on the command line.
    cond_mode: str = "channels"
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.feature_set not in BEV_FEATURE_SETS:
            raise ValueError(
                f"unknown feature_set {self.feature_set!r}; "
                f"expected one of {BEV_FEATURE_SETS}")
        if self.cond_region not in COND_REGIONS:
            raise ValueError(
                f"unknown cond_region {self.cond_region!r}; "
                f"expected one of {COND_REGIONS}")
        self.use_depth_index = self.feature_set == "v2"
        self.use_fluence = self.feature_set in ("v3", "v4")
        if self.feature_set == "v4":
            # DERIVED, like use_fluence: the set name is the single source of
            # truth, so "v4" cannot be recorded in a checkpoint while the flags
            # say something else.
            self.use_position = False        # lat_h (+0.0015), lat_w (+0.0112)
            self.use_source_distance = False  # r=1.000 with geometric_depth
            self.conditioning = False         # all three cond_* are at ~1.2x init

    def n_scalar_channels(self) -> int:
        n = 1                                            # normalised dose
        n += int(self.use_density)
        n += int(self.use_electron_density)
        n += int(self.use_cumulative_path)
        n += int(self.use_geometric_depth)
        n += 2 * int(self.use_position)                  # lat_h, lat_w
        n += int(self.use_depth_index)
        n += int(self.use_fluence)
        n += int(self.use_source_distance)
        n += 3 * int(self.cond_as_channels())
        return n

    def cond_dim(self) -> int:
        """Width of the FiLM conditioning vector; 0 disables FiLM entirely."""
        return 3 if (self.conditioning and self.cond_mode == "film") else 0

    def cond_as_channels(self) -> bool:
        return bool(self.conditioning) and self.cond_mode == "channels"


def cumulative_path_length(rho_e: torch.Tensor, spacing_depth_mm: float) -> torch.Tensor:
    """Cumulative electron-density path length along the beam axis.

    The photon analogue of the proton's water-equivalent depth: "how much
    material has this ray already traversed". For photons the physically
    correct cumulative quantity is the attenuation integral int(mu dl), and with
    Compton dominating at this spectrum mu is proportional to ELECTRON density
    -- hence integrating rho_e, not mass density.

    Axis-aligned cumulative sum along D. Rays diverge, so this is exact only on
    the central axis; at 200 mm off-axis with SAD 1000 mm the obliquity error is
    about 2%. Pass ``cum_override`` to ``build_bev_features`` to substitute the
    engine's properly divergent ``divergent_radiological_depth`` instead.
    """
    # Trapezoid-style: half of the current voxel plus all preceding ones, so the
    # value at the entrance voxel is half a voxel of path rather than a full one.
    cum = torch.cumsum(rho_e, dim=-3) - 0.5 * rho_e
    return cum * float(spacing_depth_mm)


def build_bev_features(
    dose_bev: torch.Tensor,
    density_bev: torch.Tensor,
    hu_bev: torch.Tensor,
    depth_mm: torch.Tensor,
    lat_h_mm: torch.Tensor,
    lat_w_mm: torch.Tensor,
    *,
    fluence_bev: torch.Tensor | None = None,
    open_area_mm2: torch.Tensor | float | None = None,
    sad_mm: float = 1000.0,
    spacing_depth_mm: float = 2.0,
    cum_override: torch.Tensor | None = None,
    cfg: FeatureConfig | None = None,
) -> dict[str, torch.Tensor | None]:
    """Assemble the corrector input stack.

    All volume arguments are ``[B, 1, D, H, W]``; the three coordinate vectors
    are 1-D of length D, H, W respectively and are in mm, with ``depth_mm``
    measured FROM THE SOURCE (so it is ~SAD at isocentre) and the lateral pair
    measured from the central axis.

    Returns a dict with ``features`` ``[B, C, D, H, W]``, plus ``aperture_state``
    and ``material_id`` as integer maps and ``cond`` as ``[B, cond_dim]`` -- the
    three things the model consumes through embeddings rather than channels.
    """
    cfg = cfg or FeatureConfig()
    B = dose_bev.shape[0]
    dev, dt = dose_bev.device, dose_bev.dtype
    names: list[str] = []

    scale = dose_bev.detach().amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
    chans = [dose_bev / scale]
    names.append("dose_norm")

    density = quantize_density(density_bev) if cfg.quantize_density_to_sim else density_bev
    rho_e = relative_electron_density(density, hu_bev)

    if cfg.use_density:
        chans.append(density)
        names.append("density")
    if cfg.use_electron_density:
        chans.append(rho_e)
        names.append("electron_density")
    if cfg.use_cumulative_path:
        cum = cum_override if cum_override is not None else cumulative_path_length(
            rho_e, spacing_depth_mm)
        chans.append(cum / LENGTH_SCALE_MM)
        names.append("cum_electron_path")

    # Broadcast the 1-D axis coordinates to the volume shape.
    d_view = depth_mm.to(dev, dt).view(1, 1, -1, 1, 1)
    # The lateral axes are 1-D [size] for a shared box, or [N, size] when the
    # crop gives every control point its own window (PerSampleBevCrop). In the
    # per-sample case each sample's axis is centred on its own field, which is
    # the whole point: after the offset below, every sample sees the same
    # relative coordinates no matter what it was batched with.
    _h = lat_h_mm.to(dev, dt)
    _w = lat_w_mm.to(dev, dt)
    h_view = (_h.view(-1, 1, 1, _h.shape[-1], 1) if _h.dim() == 2
              else _h.view(1, 1, 1, -1, 1))
    w_view = (_w.view(-1, 1, 1, 1, _w.shape[-1]) if _w.dim() == 2
              else _w.view(1, 1, 1, 1, -1))
    shape = dose_bev.shape

    if cfg.use_geometric_depth:
        # Depth measured from isocentre rather than from the source, so the
        # channel is centred near zero instead of near SAD/100 = 10.
        chans.append(((d_view - sad_mm) / LENGTH_SCALE_MM).expand(shape))
        names.append("geometric_depth")
    # Centre of the supplied lateral extent. Under --bev_crop that is the field
    # centre, because compute_bev_crop boxes the aperture support and grows it
    # symmetrically. Zero when the feature is off, so the channels stay
    # measured from the central axis exactly as before.
    h_off = w_off = 0.0
    if cfg.crop_relative_position:
        if _h.dim() == 2:                       # per-sample windows
            h_off = 0.5 * (_h.amin(dim=1) + _h.amax(dim=1)).view(-1, 1, 1, 1, 1)
            w_off = 0.5 * (_w.amin(dim=1) + _w.amax(dim=1)).view(-1, 1, 1, 1, 1)
        else:
            h_off = 0.5 * float(lat_h_mm.min() + lat_h_mm.max())
            w_off = 0.5 * float(lat_w_mm.min() + lat_w_mm.max())
    if cfg.use_position:
        chans.append(((h_view - h_off) / LENGTH_SCALE_MM).expand(shape))
        chans.append(((w_view - w_off) / LENGTH_SCALE_MM).expand(shape))
        names += ["lat_h", "lat_w"]
    if cfg.use_depth_index:
        n_d = shape[-3]
        idx = torch.linspace(0.0, 1.0, n_d, device=dev, dtype=dt).view(1, 1, -1, 1, 1)
        chans.append(idx.expand(shape))
        names.append("depth_index")
    if cfg.use_source_distance:
        dist = torch.sqrt(d_view ** 2 + h_view ** 2 + w_view ** 2) / float(sad_mm)
        chans.append(dist.expand(shape))
        names.append("source_distance")
    if cfg.use_fluence:
        # The projected aperture, peak-normalised, kept CONTINUOUS so the
        # sub-voxel penumbra edge survives -- see the module docstring.
        #
        # Substituting zeros when the caller forgets to pass it is the exact
        # failure that biased every v1-vs-v2 comparison in this repo: the model
        # trains on a real field and is scored on a constant, and nothing
        # errors. Fail here instead.
        if fluence_bev is None:
            raise ValueError(
                "feature_set='v3' needs fluence_bev (the projected aperture); "
                "the engine caches it as CorrectedDoseEngine._bev_fluence. "
                "Passing None would silently feed the model a zero channel.")
        fmax = fluence_bev.detach().amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
        chans.append((fluence_bev / fmax).expand(shape))
        names.append("fluence")

    features = torch.cat([c.to(dt) for c in chans], dim=1)

    aperture_state = None
    if cfg.use_aperture:
        if fluence_bev is None:
            aperture_state = torch.full(
                (B, 1) + tuple(shape[2:]), APERTURE_OPEN, device=dev, dtype=torch.long)
        else:
            # Two labels only: the ray either passes the open aperture or it
            # does not. The fluence volume is the aperture already projected
            # along DIVERGING rays by the engine, which is the one place the
            # fan geometry genuinely matters.
            fmax = fluence_bev.detach().amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
            inside = (fluence_bev / fmax) > cfg.aperture_threshold
            aperture_state = torch.where(
                inside,
                torch.full_like(inside, APERTURE_OPEN, dtype=torch.long),
                torch.full_like(inside, APERTURE_OUTSIDE, dtype=torch.long),
            )

    material_id = material_id_from_hu(hu_bev) if cfg.use_material else None

    cond = None
    if cfg.conditioning and cfg.cond_mode != "off":
        # The two axes along which the residual SCALE error was measured to be
        # coherent: aperture area (ls_scale 0.92 small vs 1.00 large) and
        # anatomy (thorax 0.91-0.93 vs abdomen 0.99-1.00). Lung fraction is the
        # cheapest anatomy proxy that separates those two cohorts.
        if open_area_mm2 is None:
            area = torch.full((B,), 0.0, device=dev, dtype=dt)
        else:
            area = torch.as_tensor(open_area_mm2, device=dev, dtype=dt).reshape(-1)
            if area.numel() == 1 and B != 1:
                area = area.expand(B)
        rho_flat = rho_e.flatten(1)

        if cfg.cond_region == "dose":
            # ORIGINAL definition, kept so that checkpoints trained against it
            # can still be evaluated on their own inputs -- swapping the region
            # under a trained model changes two of its thirteen input channels
            # by more than their whole between-cohort range, and nothing in the
            # weights would reveal the mismatch.
            #
            # It is also WRONG, which is why it is no longer the one to train
            # on: air is rho_e ~ 0 and so passes `rho_e < cond_lung_rho`, and
            # the pencil beam deposits above the 10% cutoff outside the patient,
            # so on abdomen this measures empty space rather than anatomy.
            # Decomposed by density band, 1ABB045 reads lung_frac 0.370 of which
            # 0.354 is air and 0.015 real lung; 1ABB123 reads 0.146 with 0.144
            # air. That leaves an abdomen patient (0.247) sitting next to a
            # thorax patient (0.309) and destroys the cohort separation the
            # channel exists for.
            region = (dose_bev > 0.1 * scale).flatten(1)
            denom = region.sum(dim=1).clamp_min(1).to(dt)
        else:
            # CORRECTED: the open aperture, intersected with the patient, over
            # the depth the dose actually covers. Same three-way region gives
            # abdomen 0.005-0.026 against thorax 0.280-0.474 -- a 10.7x gap with
            # no overlap.
            #
            # The aperture is normalised PER DEPTH SLICE, not against the peak
            # of the whole volume. Fluence falls off as 1/r^2, so a single
            # global `fluence > 0.5 * max` thresholds against the value at the
            # box's source-side face and keeps only the shallow part of the
            # column, which makes the "aperture" depend on how deep the box
            # happens to be: measured on 1THB119 cp135, 16018 in-field voxels
            # in the full volume against 26910 in a crop 5x smaller, moving
            # lung_frac from 0.08 to 0.42.
            #
            # `dose_bev > 0.1*peak` supplies the DEPTH extent. The aperture is a
            # lateral mask -- a column with no extent of its own along D -- so
            # on its own it inherits whatever depth the box has. The dose term
            # is the one here that is structurally inside the crop (the crop is
            # built from the dose support at 1e-3 of peak, a strictly larger
            # set), so it makes the whole region crop-invariant by construction
            # rather than by an argument about where tissue can be. Verified
            # bit-exact across 12 control points on 4 patients.
            if fluence_bev is None:
                raise ValueError(
                    "cond_region='aperture' averages the conditioning scalars "
                    "over the projected aperture, so it needs fluence_bev; the "
                    "engine caches it as CorrectedDoseEngine._bev_fluence. "
                    "Falling back to a dose threshold would silently reinstate "
                    "the air contamination this region replaced.")
            cmax = fluence_bev.detach().amax(
                dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            region = ((fluence_bev / cmax > cfg.aperture_threshold)
                      & (rho_e >= cfg.cond_air_rho)
                      & (dose_bev > 0.1 * scale)).flatten(1)
            denom = region.sum(dim=1)
            # No fallback, deliberately. Every control point in this dataset has
            # an open aperture (0 fully closed of 40500 CPs, both splits) and it
            # points at an isocentre inside the patient, so an empty region
            # means the field missed the body or the fluence is not the one that
            # made this dose -- both bugs. Substituting a default would hand the
            # model mean_rho=0, which reads as "pure air", and it would train
            # right through it.
            if not bool((denom > 0).all()):
                bad = int((denom == 0).sum())
                raise ValueError(
                    f"conditioning region empty for {bad}/{B} samples: the open "
                    f"aperture (fluence > {cfg.aperture_threshold} of peak) "
                    f"does not intersect the patient (rho_e >= "
                    f"{cfg.cond_air_rho}) anywhere the dose reaches 10% of "
                    f"peak. The field cannot miss the body; check that "
                    f"fluence_bev belongs to this beam and that the BEV crop "
                    f"kept the anatomy.")
            denom = denom.to(dt)

        mean_rho = (rho_flat * region).sum(dim=1) / denom
        lung_frac = ((rho_flat < cfg.cond_lung_rho) & region).sum(dim=1) / denom
        cond = torch.stack([
            torch.log1p(area.clamp_min(0.0)) / 10.0,
            mean_rho,
            lung_frac,
        ], dim=1).to(dt)

    # "channels" mode: broadcast the 3 scalars over the volume and append them,
    # then drop `cond` so it CANNOT also reach FiLM. The information is
    # identical; what changes is that it now enters once, at the stem, as
    # evidence a convolution weighs against its neighbourhood -- rather than as
    # an unbounded per-channel affine applied again in every block.
    if cond is not None and cfg.cond_as_channels():
        n, _, d, h, w = features.shape
        n_cond = cond.shape[1]          # 3, or 5 with the crop-offset scalars
        features = torch.cat(
            [features,
             cond.view(n, n_cond, 1, 1, 1).expand(n, n_cond, d, h, w).to(features.dtype)],
            dim=1,
        )
        names = names + ["cond_log_area", "cond_mean_rho", "cond_lung_frac"]
        cond = None

    cfg.names = names
    return {
        "features": features,
        "aperture_state": aperture_state,
        "material_id": material_id,
        "cond": cond,
        "scale": scale,
        "names": names,
    }
