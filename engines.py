"""
Main class for radiotherapy dose calculation using pencil beam convolution and beam-wise rotation.

This class orchestrates the pipeline for dose calculation, including preprocessing, fluence modeling,
kernel generation, convolution, and geometric rotation of dose volumes. It supports batched inputs and
multiple beams, and can optionally perform upsampling and debugging visualizations.
"""
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pydosert.layers.FluenceMapLayer import FluenceMapLayer
from pydosert.layers.FluenceVolumeLayer import FluenceVolumeLayer
import math

from pydosert.engine.multislab_engine import (
    divergent_radiological_depth, lateral_scatter_correction, _gaussian_blur_lateral,
)
from terma_scaling import TermaScalingLayer


# Density bin edges for the fine lateral-scatter model. pydosert's built-in
# lateral_scatter_correction uses three bins with FIXED representative densities
# (0.30 / 0.60 / 0.85), so every voxel below rho=0.45 is treated as rho=0.30.
# Real lung in this dataset has median rho ~0.135 (p10 0.08), where the physical
# electron-range scaling (1/rho - 1) is 6.4, not 2.3 — the built-in model
# under-spreads deep lung by ~3x, which is most of the thorax error.
_LAT_BIN_EDGES = (0.001, 0.08, 0.14, 0.22, 0.35, 0.55, 0.92)


def _lat_bin_table(edges=_LAT_BIN_EDGES):
    """(lo, hi, rho_rep) per bin. rho_rep is chosen so that 1/rho_rep equals the
    mean of 1/rho over the bin (uniform in rho), i.e. rho_rep = (b-a)/ln(b/a)."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append((a, b, (b - a) / math.log(b / a)))
    return tuple(out)


_LAT_BINS = _lat_bin_table()


def lateral_scatter_correction_fine(dose_bev, bev_density, lat_sigma_mm, res_lat_mm,
                                    lat_cap_mm=None, bins=_LAT_BINS):
    """Density-scaled lateral scatter, same physics as pydosert's version but with
    density resolved finely enough to distinguish deep lung from soft tissue.

    What it does, concretely: dose sitting in low-density voxels is taken out,
    blurred sideways (perpendicular to the beam), and added back. Nothing is
    created or destroyed — it is redistributed. That is what real electron
    transport does in lung and what a fixed-width pencil-beam kernel cannot do.

    ``lat_sigma_mm`` is either
      * a scalar - the extra lateral sigma for a bin is ``sigma * (1/rho - 1)``,
        i.e. one global parameter and the 1/rho electron-range scaling; or
      * a sequence of ``len(bins)`` values - the extra sigma in millimetres for
        each density bin directly, no 1/rho assumption. Six free parameters
        instead of one, which lets the fit discover the real density dependence
        rather than imposing 1/rho (that law comes from CSDA range in a uniform
        medium and is only approximate for a finite field in a finite lung).

    Dose-conserving: each bin's dose is removed and re-added blurred."""
    per_bin = not isinstance(lat_sigma_mm, (int, float))
    if per_bin and len(lat_sigma_mm) != len(bins):
        raise ValueError(f"lat_sigma_mm has {len(lat_sigma_mm)} entries, expected {len(bins)}")
    out = dose_bev
    for i, (lo, hi, rho) in enumerate(bins):
        m = ((bev_density >= lo) & (bev_density < hi)).to(dose_bev.dtype)
        if m.sum() == 0:
            continue
        extra_sigma_mm = (float(lat_sigma_mm[i]) if per_bin
                          else lat_sigma_mm * (1.0 / rho - 1.0))
        if lat_cap_mm is not None:
            extra_sigma_mm = min(extra_sigma_mm, lat_cap_mm)
        if extra_sigma_mm <= 1e-6:
            continue
        d_bin = dose_bev * m
        out = out - d_bin + _gaussian_blur_lateral(d_bin, extra_sigma_mm / res_lat_mm)
    return out
from pydosert.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydosert.layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from pydosert.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydosert.layers.BeamRotationLayer import BeamRotationLayer
from pydosert.data import MachineConfig, Beam, BeamSequence
from pydosert.geometry.rotations import rotate_2d_images

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-excitation channel attention.

    Cheap (a couple of 1x1x1 convs on the globally-pooled descriptor) and
    reliably helps dense-prediction nets recalibrate channel responses.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class BottleneckSelfAttention3D(nn.Module):
    """Multi-head self-attention over the flattened bottleneck volume.

    Only ever applied at the U-Net bottleneck where the spatial extent is
    smallest (input / 2**depth), so the N^2 attention cost is bounded.
    Uses torch's memory-efficient scaled_dot_product_attention. The output
    projection is zero-initialised so the block starts as an identity and
    doesn't disturb the calibrated baseline + residual model early on.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        # channels must be divisible by num_heads; fall back gracefully.
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W
        hc = C // self.num_heads
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, hc, N)
        q, k, v = qkv.unbind(1)                       # B, heads, hc, N
        q = q.transpose(-2, -1)                       # B, heads, N, hc
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        out = F.scaled_dot_product_attention(q, k, v)  # B, heads, N, hc
        out = out.transpose(-2, -1).reshape(B, C, D, H, W)
        return x + self.proj(out)


class _ApertureMaskContext:
    """Carries one forward's aperture mask to every masked norm in a model.

    The mask is a property of the control point, not of any one layer, and it
    has to reach ~14 norm layers at 5 different resolutions. Passing it down
    the call chain would mean changing the signature of every block; instead
    the model parks it here for the duration of its forward and the norms read
    it. `pooled` caches the per-resolution downsample so the pyramid is built
    once per forward rather than once per layer.
    """

    __slots__ = ("mask", "pooled", "any_empty")

    def __init__(self):
        self.mask = None
        self.pooled = {}
        self.any_empty = False

    def set(self, mask):
        self.mask = mask
        self.pooled = {}
        # Emptiness is decided ONCE per forward, here, rather than inside every
        # masked norm. Reading it on the host is a device sync, and there are 22
        # of those layers -- measured at 3.06 ms of a 72 ms forward purely in
        # stalls. Max-pooling preserves emptiness exactly (an empty window stays
        # empty, a non-empty one stays non-empty), so the flag computed at full
        # resolution is valid at every level of the pyramid.
        self.any_empty = bool((mask.flatten(1).sum(dim=1) <= 0).any())

    def clear(self):
        self.mask = None
        self.pooled = {}
        self.any_empty = False

    def for_shape(self, shape):
        """The mask at one spatial resolution, as float [N, 1, D, H, W]."""
        if self.mask is None:
            return None
        key = tuple(shape)
        got = self.pooled.get(key)
        if got is None:
            m = self.mask
            if tuple(m.shape[-3:]) != key:
                # MAX pool, not average: a coarse voxel is in-field if ANY of
                # the fine voxels under it is. Average pooling would fade the
                # mask out at depth and quietly turn the statistics back into
                # whole-volume ones, which is the thing being fixed.
                m = F.adaptive_max_pool3d(m, key)
            got = m
            self.pooled[key] = got
        return got


class MaskedGroupNorm3d(nn.GroupNorm):
    """GroupNorm whose statistics come only from inside the aperture.

    Plain GroupNorm averages over the WHOLE box. Under --bev_crop the box is
    the field plus a 50 mm margin, so the in-field fraction changes with the
    field size -- and the field size changes at every control point as the MLC
    moves. Two control points with identical physics inside the aperture then
    get different means and variances purely because one has more margin
    around it, and the network sees a shifted, rescaled version of the same
    input. The crop makes this worse, not better: the smaller the box, the more
    the statistics swing with the aperture.

    Masking fixes the reference. Mean and variance are accumulated over
    in-field voxels only, so they describe the dose being corrected rather than
    the box it was cut from. The normalisation is then applied to EVERY voxel,
    including the margin -- the margin still gets normalised, it just no longer
    gets a vote on the statistics.

    Falls back to standard GroupNorm when no mask is set (no aperture channel,
    or a closed MLC whose mask is empty), so the layer is always well defined.
    """

    def __init__(self, num_groups: int, num_channels: int, ctx):
        super().__init__(num_groups, num_channels)
        # Wrapped in a tuple so nn.Module does not try to register it.
        self._ctx = (ctx,)

    def forward(self, x):
        ctx = self._ctx[0]
        m = ctx.for_shape(x.shape[-3:]) if ctx is not None else None
        # No host read here: the context decided this once for the whole
        # forward. An empty mask has nothing to key on, so fall back to
        # whole-box statistics rather than dividing by zero.
        if m is not None and ctx.any_empty:
            m = None

        N, C = x.shape[:2]
        G = self.num_groups
        xg = x.reshape(N, G, C // G, *x.shape[-3:])
        dims = (2, 3, 4, 5)

        # DELIBERATELY NOT nn.GroupNorm.forward when there is no mask.
        # torch.autocast promotes F.group_norm to fp32, so a plain GroupNorm
        # runs every norm layer in fp32 and hands the next conv an fp32
        # activation to cast back down -- roughly double the memory traffic on
        # 22 layers, which showed up as the `group` arm reporting ~2x the
        # runtime of `group_masked` for reasons that had nothing to do with
        # normalisation. This path stays in x.dtype (statistics still
        # accumulate in fp32), so every norm arm is arithmetically comparable
        # and only the POOLING WIDTH differs, which is what the ablation varies.
        if m is None:
            mg = None
            count = torch.full((), float(xg[0, 0].numel()),
                               device=x.device, dtype=torch.float32)
        else:
            mg = m.reshape(N, 1, 1, *m.shape[-3:]).to(x.dtype)
            count = mg.sum(dim=dims, keepdim=True) * (C // G)

        # Sums of x and x^2 rather than a second pass over (xg - mean).
        # The literal two-pass form materialises (xg - mean), its square, the
        # masked product and the normalised tensor at full resolution and
        # retains every one of them for backward -- the OOM described in
        # _DilatedRefine3D below. Here only `xm` is a full-size temporary, and
        # the normalisation collapses into a single fused multiply-add whose
        # scale and shift are per-group scalars.
        # Centred on a detached per-group origin. Raw sums of x and x^2 would
        # halve the temporaries too, but dose is a small spread on a large
        # offset and E[x^2] - E[x]^2 then cancels to ~1e-2 relative in fp32.
        # Subtracting K first costs nothing (it carries no graph) and keeps the
        # cancellation-free accuracy of the two-pass form.
        acc = torch.float32 if x.dtype not in (torch.float64,) else torch.float64
        K = xg.mean(dim=dims, keepdim=True).detach()
        d = xg - K
        dm = d if mg is None else d * mg
        cf = count.to(acc)
        s1 = dm.sum(dim=dims, keepdim=True, dtype=acc)
        s2 = (d * dm).sum(dim=dims, keepdim=True, dtype=acc)
        mean_c = s1 / cf
        mean = mean_c + K.to(acc)
        var = (s2 / cf - mean_c * mean_c).clamp_min(0.0)
        scale = torch.rsqrt(var + self.eps)
        if self.affine:
            scale = scale * self.weight.view(1, G, C // G, 1, 1, 1)
            shift = self.bias.view(1, G, C // G, 1, 1, 1) - mean * scale
        else:
            shift = -mean * scale
        out = torch.addcmul(shift.to(x.dtype), xg, scale.to(x.dtype))
        return out.reshape(N, C, *x.shape[-3:])


def _norm3d(kind: str, channels: int, ctx=None) -> nn.Module:
    """BatchNorm or GroupNorm for the v1 trunk.

    BatchNorm is the historical default and is a poor fit here. A batch is
    `batched_cp_size` control points from the SAME beam -- 2 degrees apart and
    near-duplicates -- so the batch statistics are estimated from four
    correlated samples, not four independent ones. Worse, training runs 3 beams
    per patient and validation runs 1, so the running statistics are collected
    on one distribution and applied to another. GroupNorm normalises per sample
    and has neither failure mode.

    "group_masked" additionally restricts the statistics to the aperture; see
    MaskedGroupNorm3d for why the box makes them field-dependent otherwise.
    """
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind in ("group", "group_masked"):
        groups = max(1, min(8, channels // 4))
        while channels % groups:
            groups -= 1
        # Both go through MaskedGroupNorm3d; `group` simply gets no mask
        # context, so its statistics come from the whole box. Same arithmetic,
        # same dtype -- see the autocast note in MaskedGroupNorm3d.forward.
        return MaskedGroupNorm3d(groups, channels,
                                 ctx if kind == "group_masked" else None)
    if kind == "instance":
        # One group PER CHANNEL is InstanceNorm with affine. Same class again,
        # so parameter shapes and arithmetic match the other two arms exactly.
        return MaskedGroupNorm3d(channels, channels, None)
    if kind in ("none", None):
        # No normalisation at all. Worth having as a control: the masked norm
        # keys its statistics on the in-aperture voxels, which are under 1% of
        # the volume when the corrector is not cropped, so it is being asked to
        # normalise a whole feature map from a very small and very variable
        # sample.
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


class SeparableConv3d(nn.Module):
    """Factorised 3D convolution: (1,k,k) LATERAL, then (kd,1,1) along DEPTH.

    The BEV stack is ``[B, G, D, H, W]`` with D the beam depth axis (see
    ``CorrectedDoseEngine.source_distance_bev``), so this split is not an
    arbitrary factorisation -- the two factors are different physics. The
    lateral factor is the scatter kernel in the transverse plane, short-range
    and roughly isotropic in-plane. The depth factor is attenuation and
    build-up along the ray: smooth, monotonic, and long-range.

    Cost per voxel is ``9*Cin*Cout + kd*Cout*Cout`` against ``27*Cin*Cout`` for
    a dense 3x3x3, i.e. 14/27 at Cin==Cout with kd=5. That is what pays for the
    width: base 16 with the two finest levels separable costs 1.25x the shipped
    base-8 dense model while carrying 3.9x the parameters, where widening the
    dense model to base 16 costs 2.11x. It also makes a LONGER depth kernel
    almost free (kd=5 -> 7 is +2*Cout^2 per voxel), which is the cheapest way
    to buy reach on the one axis where the physics is genuinely long-range.

    A ReLU sits between the two factors (R(2+1)D-style) rather than composing
    them into a single linear operator, so the block is strictly more
    expressive than the rank-1 factorisation of a dense kernel.
    """

    def __init__(self, in_channels, out_channels, lateral_kernel=3,
                 depth_kernel=5, dilation=1, depth_dilation=None,
                 norm="batch", norm_ctx=None):
        super().__init__()
        dd = dilation if depth_dilation is None else depth_dilation
        pad_l = dilation * (lateral_kernel // 2)
        pad_d = dd * (depth_kernel // 2)
        self.lateral = nn.Conv3d(
            in_channels, out_channels, (1, lateral_kernel, lateral_kernel),
            padding=(0, pad_l, pad_l), dilation=(1, dilation, dilation),
            bias=False)
        # NO normalisation between the two factors, only an in-place ReLU.
        #
        # An inner MaskedGroupNorm3d here is what OOM'd job 561517 -- an 80 GB
        # H100, in the FORWARD pass of the first batch, at batched_cp_size 1,
        # before a single profile line. It died at `out = out * w + b`
        # (engines.py:259) inside the refine branch.
        #
        # The reason is that MaskedGroupNorm3d is hand-written broadcasting
        # arithmetic, not a fused kernel: it materialises (xg - mean), its
        # square, the masked product, the normalised tensor and the affine
        # output -- roughly five full-resolution tensors, every one of them
        # retained for backward. Putting one INSIDE each factorised conv
        # doubles the count of those at full resolution, so factorising halved
        # the multiply-adds while nearly doubling the stored activations in the
        # most expensive branch. Compute and memory move in opposite directions
        # here, which is exactly what a FLOP-based budget cannot see.
        #
        # The block still ends in the outer norm that Double3DConv and
        # _DilatedRefine3D apply, so there is one norm per logical convolution
        # -- the same count the dense model has -- and the ReLU between the
        # factors keeps this strictly more expressive than a rank-1
        # factorisation of a dense kernel.
        self.act = nn.ReLU(inplace=True)
        self.depth = nn.Conv3d(
            out_channels, out_channels, (depth_kernel, 1, 1),
            padding=(pad_d, 0, 0), dilation=(dd, 1, 1))

    def forward(self, x):
        return self.depth(self.act(self.lateral(x)))


class Double3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, residual=False,
                 norm="batch", norm_ctx=None, separable=False,
                 lateral_kernel=3, depth_kernel=5):
        super().__init__()

        def conv(a, b):
            if not separable:
                return nn.Conv3d(a, b, kernel_size=3, padding=1)
            return SeparableConv3d(a, b, lateral_kernel=lateral_kernel,
                                   depth_kernel=depth_kernel,
                                   norm=norm, norm_ctx=norm_ctx)

        layers = [
            conv(in_channels, out_channels),
            _norm3d(norm, out_channels, norm_ctx),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels),
            _norm3d(norm, out_channels, norm_ctx),
            nn.ReLU(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock3D(out_channels))
        self.conv = nn.Sequential(*layers)
        self.residual = residual
        if residual:
            self.proj = (nn.Identity() if in_channels == out_channels
                         else nn.Conv3d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        out = self.conv(x)
        return out + self.proj(x) if self.residual else out


def pad_to_multiple(x, multiple=16):
    """`multiple` may be a scalar or a per-axis (D, H, W) triple.

    Anisotropic pooling needs the triple: with a (2,1,1) pool the depth axis
    must divide by 2**levels while H and W must not be padded at all, and a
    scalar would silently pad every axis to the largest requirement."""
    if isinstance(multiple, int):
        multiple = (multiple, multiple, multiple)
    d, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    md, mh, mw = multiple
    pd = (md - d % md) % md
    ph = (mh - h % mh) % mh
    pw = (mw - w % mw) % mw
    x = F.pad(x, (0, pw, 0, ph, 0, pd))
    return x, (d, h, w)


class _UNet3D(nn.Module):
    """Plain 3D U-Net trunk used by the dose correction model."""

    def __init__(self, in_channels: int, base_channels: int, depth: int = 4,
                 attention: str = "none", residual: bool = False,
                 norm: str = "batch", norm_ctx=None, sep_levels: int = 0,
                 lateral_kernel: int = 3, depth_kernel: int = 5,
                 pool_kernels=None, width_growth: float = 2.0,
                 downsample_mode: str = "maxpool",
                 activation_checkpointing: bool = False):
        """attention: "none" | "se" (SE in every conv block) | "sa"
        (SE blocks + a self-attention layer at the bottleneck). residual: skip
        connection around each double-conv block (helps deeper nets / fine residuals).

        sep_levels: how many of the FINEST levels use SeparableConv3d instead of
        a dense 3x3x3, counted from level 0 (full resolution) and applied
        symmetrically to the matching decoder levels. 0 reproduces the dense
        model exactly, key-for-key, which is what every existing checkpoint
        needs. Only the fine levels are worth factorising: a level at 2**i
        downsample holds 8**i fewer voxels, so level 0 alone is 54% of this
        trunk's multiply-adds and everything below level 1 is nearly free.
        The bottleneck stays dense for the same reason -- it is where 3D
        mixing is most useful and where it costs least."""
        super().__init__()
        self.depth = depth
        self.attention = attention
        use_se = attention in ("se", "sa")
        b = base_channels
        # ANISOTROPIC POOLING. BEV is [B,C,D,H,W] with D the beam depth, and the
        # two directions do not carry the same spatial frequency: laterally the
        # penumbra is ~5 mm, i.e. 2.5 voxels at 2 mm, while along depth the dose
        # is build-up followed by a smooth exponential. Pooling both by 16x, as
        # (2,2,2) x 4 does, throws away the lateral detail and keeps resolution
        # the depth axis does not need.
        #
        # This is not a speculative concern: _DilatedRefine3D exists precisely
        # to work around it ("the U-Net pools depth times, which is exactly what
        # destroys the high-frequency dose detail"), and that 24k-parameter
        # branch out-contributes the 1.3M-parameter trunk by 3.4x in the
        # ablation. Fixing the pooling addresses the cause.
        #
        # width_growth is coupled to it and cannot be left at 2.0 blindly:
        # doubling the channels pays for an 8x voxel reduction, so a (2,1,1)
        # pool that only halves the voxels makes each level 2x MORE expensive
        # than the one above rather than 2x cheaper.
        self.pool_kernels = [tuple(k) for k in (pool_kernels
                             or [(2, 2, 2)] * depth)]
        if len(self.pool_kernels) != depth:
            raise ValueError(f"pool_kernels needs {depth} entries, "
                             f"got {len(self.pool_kernels)}")
        widths = [max(1, int(round(b * (width_growth ** i))))
                  for i in range(depth + 1)]
        if downsample_mode not in ("maxpool", "strideconv"):
            raise ValueError("downsample_mode must be 'maxpool' or 'strideconv', "
                             f"got {downsample_mode!r}")
        self.downsample_mode = downsample_mode
        self.activation_checkpointing = bool(activation_checkpointing)
        self._norm_ctx = norm_ctx
        # Encoder
        self.sep_levels = int(sep_levels)
        sep_kw = dict(lateral_kernel=lateral_kernel, depth_kernel=depth_kernel)
        self.encs = nn.ModuleList()
        prev = in_channels
        for i, w in enumerate(widths[:-1]):
            self.encs.append(Double3DConv(prev, w, use_se=use_se, residual=residual,
                                          norm=norm, norm_ctx=norm_ctx,
                                          separable=i < self.sep_levels, **sep_kw))
            prev = w
        if downsample_mode == "maxpool":
            self.pools = nn.ModuleList(
                [nn.MaxPool3d(k) for k in self.pool_kernels])
        else:
            # Learned anti-alias/downsampling. Axes whose stride is one get a
            # 1-wide kernel, so (2,1,1) mixes only along beam depth and keeps
            # the lateral samples exactly where they are. Channels are kept
            # constant here; the following encoder block performs the width
            # change, matching the historical pool -> block ordering.
            self.pools = nn.ModuleList([
                nn.Conv3d(widths[i], widths[i],
                          kernel_size=tuple(2 * s - 1 for s in k),
                          stride=k, padding=tuple(s - 1 for s in k),
                          bias=False)
                for i, k in enumerate(self.pool_kernels)
            ])
        self.bottleneck = Double3DConv(widths[-2], widths[-1], use_se=use_se,
                                       residual=residual, norm=norm, norm_ctx=norm_ctx)
        self.bottleneck_attn = (
            BottleneckSelfAttention3D(widths[-1]) if attention == "sa" else None
        )
        # Decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(depth, 0, -1):
            # Mirror the pool that produced this level, so the upsample undoes
            # exactly the reduction the encoder applied on each axis.
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=self.pool_kernels[i - 1],
                            mode='trilinear', align_corners=False),
                nn.Conv3d(widths[i], widths[i - 1], kernel_size=1),
            ))
            # This block OUTPUTS level i-1, so it runs at level i-1's resolution
            # and is factorised on the same test the encoder uses.
            # The decoder block consumes up(h) PLUS the skip, and up() has
            # already projected to widths[i-1], so its input is 2*widths[i-1] --
            # NOT widths[i]. Those coincide only at width_growth exactly 2,
            # which is why the old form worked and why any other growth factor
            # failed with a channel mismatch.
            self.decs.append(Double3DConv(2 * widths[i - 1], widths[i - 1], use_se=use_se,
                                          residual=residual, norm=norm, norm_ctx=norm_ctx,
                                          separable=(i - 1) < self.sep_levels, **sep_kw))
        self.out_channels = widths[0]
        # Per-axis divisibility the input must satisfy: the product of every
        # pool along that axis.
        self.pad_multiple = tuple(
            math.prod(k[a] for k in self.pool_kernels) for a in range(3))

    def _checkpoint(self, fn, *inputs):
        """Checkpoint a block without losing masked-normalisation context.

        DoseCorrectionModel clears its mutable aperture-mask context when the
        ordinary forward returns. Checkpoint recomputation happens later,
        during backward; without restoring the mask it would silently run as
        plain GroupNorm and compute different gradients.
        """
        if not (self.training and self.activation_checkpointing and
                torch.is_grad_enabled()):
            return fn(*inputs)
        ctx = self._norm_ctx
        mask = None if ctx is None else ctx.mask

        def recompute(*xs):
            previous = None if ctx is None else ctx.mask
            if ctx is not None:
                if mask is None:
                    ctx.clear()
                else:
                    ctx.set(mask)
            try:
                return fn(*xs)
            finally:
                if ctx is not None:
                    if previous is None:
                        ctx.clear()
                    else:
                        ctx.set(previous)

        return checkpoint(recompute, *inputs, use_reentrant=False)

    def forward(self, x, return_pyramid: bool = False):
        """``return_pyramid`` additionally returns every decoder level.

        The pyramid is COARSEST-FIRST: entry ``j`` has spatial downsample
        factor ``2 ** (depth - 1 - j)`` and ``widths[depth - 1 - j]`` channels,
        so the last entry is the full-resolution output that is also the return
        value. Deep supervision hangs its aux heads off the coarse entries.
        """
        skips = []
        h = x
        for enc, pool in zip(self.encs, self.pools):
            h = self._checkpoint(enc, h)
            skips.append(h)
            h = pool(h)
        h = self._checkpoint(self.bottleneck, h)
        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)
        pyramid = []
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            # Put upsample + concatenation INSIDE the checkpoint. Otherwise the
            # full-resolution concatenated tensor becomes a checkpoint input
            # and must stay resident, forfeiting much of the memory saving.
            def decode(low, lateral, _up=up, _dec=dec):
                return _dec(torch.cat([_up(low), lateral], dim=1))
            h = self._checkpoint(decode, h, skip)
            if return_pyramid:
                pyramid.append(h)
        return (h, pyramid) if return_pyramid else h


class _DilatedRefine3D(nn.Module):
    """Full-resolution dilated-conv refinement branch (no pooling).

    The U-Net pools `depth` times, which is exactly what destroys the
    high-frequency dose detail (penumbra, density interfaces) the model
    is meant to recover. This parallel branch keeps full spatial
    resolution and grows its receptive field via dilation instead of
    pooling, so it can sharpen detail the trunk smears. Its final conv
    is zero-initialised so the whole branch starts as a no-op and can't
    destabilise the calibrated baseline early in training.
    """

    def __init__(self, in_channels, hidden=16, dilations=(1, 2, 4, 8), norm="batch",
                 norm_ctx=None, separable=False, lateral_kernel=3, depth_kernel=5,
                 depth_dilations=None):
        """separable: factorise each body conv into lateral + depth.

        This branch is the single most expensive thing in the corrector --
        four DENSE full-resolution 3x3x3 convs, 60% of the model's multiply-adds
        for 1.8% of its parameters -- so it is the first place factorisation
        pays. ``depth_dilations`` additionally lets the depth factor reach
        further than the lateral one (e.g. (1, 3, 9, 27) against a lateral
        (1, 2, 4, 8)), which costs Cout^2 per step instead of 9*Cout^2 and is
        the cheap way to cover attenuation, the one genuinely long-range axis.
        """
        super().__init__()
        if depth_dilations is None:
            depth_dilations = dilations
        if len(depth_dilations) != len(dilations):
            raise ValueError(
                f"depth_dilations has {len(depth_dilations)} entries but "
                f"dilations has {len(dilations)}; they index the same layers.")
        layers = []
        prev = in_channels
        for d, dd in zip(dilations, depth_dilations):
            if separable:
                layers += [
                    SeparableConv3d(prev, hidden, lateral_kernel=lateral_kernel,
                                    depth_kernel=depth_kernel, dilation=d,
                                    depth_dilation=dd, norm=norm, norm_ctx=norm_ctx),
                    _norm3d(norm, hidden, norm_ctx),
                    nn.ReLU(inplace=True),
                ]
            else:
                layers += [
                    nn.Conv3d(prev, hidden, kernel_size=3, padding=d, dilation=d),
                    _norm3d(norm, hidden, norm_ctx),
                    nn.ReLU(inplace=True),
                ]
            prev = hidden
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv3d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.body(x))


class LateralTTA(nn.Module):
    """Average a corrector's output over lateral mirror symmetries.

    BEV axes are (D, H, W): D is depth FROM the source and is directional, so
    it is never flipped. H (patient z / MLC leaf axis) and W (leaf ends) are
    both symmetric about the central axis -- the beam diverges symmetrically
    and lateral scatter is isotropic in-plane -- so mirroring the whole input
    stack and un-mirroring the prediction should be a no-op for a model that
    has learned the symmetry.

    That makes this a probe as much as an accuracy trick. A model that has
    learned the physics gains nothing; a gain means the network is producing
    orientation-dependent answers to an orientation-independent question, i.e.
    it is expressivity- or data-limited rather than information-limited.

    Only valid for correctors whose inputs are unsigned scalar fields. v1
    qualifies (dose, density, radiological depth). The v2 stack does NOT --
    it carries signed lateral position channels that would need negating, so
    this wrapper refuses anything that is not a plain tensor-in/tensor-out
    module.
    """

    FLIPS = ((), (-1,), (-2,), (-2, -1))

    def __init__(self, model: nn.Module, flips=None, iso_centered: bool = True):
        super().__init__()
        self.model = model
        self.flips = tuple(self.FLIPS if flips is None else flips)
        # torch.flip reflects about the ARRAY centre, but the physical symmetry
        # axis is the beam's central axis, i.e. the isocentre. Those differ by
        # 1-2 voxels here, and a 2-voxel offset becomes a 4-voxel displacement
        # after mirroring -- 8 mm across a penumbra only ~5 mm wide. Reflecting
        # about the isocentre instead: out[i] = in[2c - i], which is a flip
        # followed by a roll of 2c-(N-1).
        self.iso_centered = bool(iso_centered)
        self._centers = None            # (h_vox, w_vox), set per beam by the engine

    def set_centers(self, h_vox: float, w_vox: float) -> None:
        self._centers = (float(h_vox), float(w_vox))

    def _reflect(self, x, dims):
        if not dims:
            return x
        y = torch.flip(x, dims)
        if not (self.iso_centered and self._centers is not None):
            return y
        for d in dims:
            c = self._centers[0] if d == -2 else self._centers[1]
            shift = int(round(2.0 * c - (x.shape[d] - 1)))
            if shift:
                y = torch.roll(y, shifts=shift, dims=d)
        return y

    def forward(self, x):
        acc = None
        for dims in self.flips:
            out = self.model(self._reflect(x, dims))
            out = self._reflect(out, dims)      # reflection is its own inverse
            acc = out if acc is None else acc + out
        return acc / len(self.flips)


class DoseCorrectionModel(nn.Module):
    """Residual dose corrector with a multiplicative gain + additive residual head.

    The network sees BEV dose along with auxiliary channels (rotated density,
    optional radiological depth / log dose) and predicts:
        corrected = dose * (1 + gain) + residual
    where ``gain`` is bounded via tanh (~±max_gain) and ``residual`` is a
    relatively unconstrained additive term. Splitting the two heads decouples
    global scaling errors (heterogeneity) from local kernel-shape errors,
    which empirically tend to dominate in different anatomies (abdomen vs
    thorax respectively).

    The model accepts ``[B, C, D, H, W]`` tensors. The first channel MUST be
    the dose to correct; remaining channels are auxiliary features.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 8,
        depth: int = 4,
        max_gain: float = 0.5,
        use_gain: bool = True,
        attention: str = "none",
        refine: bool = False,
        residual: bool = False,
        norm: str = "batch",
        # Bound the additive head:
        #   D_hat = relu(D * (1 + a*tanh(g)) + alpha * s * tanh(r))
        # with s the detached per-sample peak. Unbounded by default, which is
        # what v1 shipped with -- and what blew up job 556220 at epoch 4, train
        # loss 0.0064 -> 0.0144. The v2 loss is peak-normalised, so a low-peak
        # control point yields a very large normalised gradient; with an
        # unbounded residual head, no relu and no gradient clipping, one step
        # is enough. --v1_bounded_residual turns this on and the best run uses
        # it; it is the guard rail, not merely a clip.
        bounded_residual: bool = False,
        stem_stride: int = 1,
        refine_mode: str = "parallel",
        additive_scale_frac: float = 0.05,
        # Learned embedding of the Geant4 composition family (86 classes,
        # materials_v2.material_id_from_hu). 0 disables it.
        #
        # This carries what the electron-density channel cannot. rho_e is
        # density x (Z/A), a single scalar; but roughly half the fluence at this
        # spectrum is above the 1.022 MeV pair threshold and pair production is
        # Z-dependent, so bone has a residual composition effect beyond rho_e.
        #
        # WHERE THE RESOLUTION IS, measured off GEANT4_HU_BOUNDS: 75 of the 86
        # families are bone (HU >= 140, in 20 HU steps), 9 are soft tissue, and
        # ONE spans -950..-90 HU -- lung parenchyma and fat together. So the
        # expected signature is a BONE effect.
        #
        # In lung the label is constant while HU swings -950..-500: what varies
        # there is DENSITY, not composition, and density already reaches the
        # model through the density and electron_density channels. The one
        # thorax boundary this embedding can express is air (family 0, rel Z/A
        # 0.900) against lung/fat (family 1, 0.992) at -950 HU -- airways, the
        # trachea, and the body-air interface. A thorax gain beyond that is
        # something else and is worth chasing before it is credited here.
        material_embedding_dim: int = 0,
        n_materials: int = 86,
        pool_kernels=None,
        width_growth: float = 2.0,
        # Aux heads on the coarse decoder levels. Training-only, but they add
        # parameters, so this changes the state_dict and must be recorded in the
        # checkpoint's model_config.
        deep_supervision: bool = False,
        # Anisotropic factorisation of the convolutions. BEV is [B,C,D,H,W]
        # with D the beam depth, so "lateral" and "depth" are different
        # physics, not an arbitrary axis split -- see SeparableConv3d.
        #
        # sep_levels counts from the FINEST trunk level. 0 is the dense model
        # every existing checkpoint was fitted with and reproduces its
        # state_dict key-for-key, so this is safe to default on.
        sep_levels: int = 0,
        lateral_kernel: int = 3,
        depth_kernel: int = 5,
        separable_refine: bool = False,
        # Tie the refine branch's output to the control point's dose peak, the
        # same way the trunk's residual already is. WITHOUT this the branch is
        # the one term in the correction that is scale-blind: its first norm
        # destroys the dose magnitude, and its output is added to the dose in
        # ABSOLUTE units. Measured on the e18 checkpoint, quartering the dose
        # quarters the trunk residual (x0.31) but leaves the refine output
        # essentially where it was (x0.76). Smaller apertures mean lower peak
        # dose per control point, so a scale-blind branch over-corrects exactly
        # where the test set differs from training.
        #
        # This changes the OUTPUT ALGEBRA and not one tensor, so a checkpoint
        # served under the wrong value loads clean and returns a wrong dose.
        # It rides in model_config for that reason.
        refine_scale_to_peak: bool = False,
        refine_hidden: int = 16,
        refine_dilations=(1, 2, 4, 8),
        refine_depth_dilations=None,
        downsample_mode: str = "maxpool",
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.max_gain = max_gain
        self.use_gain = use_gain

        self.bounded_residual = bool(bounded_residual)
        self.additive_scale_frac = float(additive_scale_frac)
        self.refine_scale_to_peak = bool(refine_scale_to_peak)

        self.material_embedding_dim = max(0, int(material_embedding_dim))
        self.n_materials = int(n_materials)
        self.material_embedding = (
            nn.Embedding(self.n_materials, self.material_embedding_dim)
            if self.material_embedding_dim > 0 else None
        )
        # The embedding is concatenated to the scalar stack, so BOTH the trunk
        # and the full-resolution refine branch see the wider input.
        trunk_in = in_channels + self.material_embedding_dim

        # One context per model, shared by every masked norm inside it.
        self.norm_kind = norm
        self._norm_ctx = _ApertureMaskContext() if norm == "group_masked" else None
        self.trunk = _UNet3D(trunk_in, base_channels, depth=depth,
                             attention=attention, residual=residual, norm=norm,
                             norm_ctx=self._norm_ctx, sep_levels=sep_levels,
                             lateral_kernel=lateral_kernel,
                             depth_kernel=depth_kernel,
                             pool_kernels=pool_kernels,
                             width_growth=width_growth,
                             downsample_mode=downsample_mode,
                             activation_checkpointing=activation_checkpointing)
        b_out = self.trunk.out_channels

        # "parallel"   : the refiner reads the RAW input and its output is ADDED
        #                to the trunk's correction. Neither branch sees the
        #                other's answer, so both must independently infer the
        #                correction and their SUM has to be right.
        # "sequential" : the refiner reads the raw input PLUS the trunk's
        #                correction (peak-normalised, as an extra channel) and
        #                predicts a residual on top -- a cascade, so it can
        #                correct what the trunk got wrong instead of guessing
        #                the same quantity twice.
        if refine_mode not in ("parallel", "sequential"):
            raise ValueError(f"refine_mode must be parallel|sequential, got {refine_mode!r}")
        self.refine_mode = refine_mode
        _refine_in = trunk_in + (1 if refine_mode == "sequential" else 0)
        self.refine = (_DilatedRefine3D(
                           _refine_in, hidden=refine_hidden,
                           dilations=tuple(refine_dilations), norm=norm,
                           norm_ctx=self._norm_ctx, separable=separable_refine,
                           lateral_kernel=lateral_kernel,
                           depth_kernel=depth_kernel,
                           depth_dilations=(tuple(refine_depth_dilations)
                                            if refine_depth_dilations else None))
                       if refine else None)
        # Run the POOLING TRUNK at 1/stem_stride resolution and resample its
        # correction back up. The trunk's first level is ~75% of its activation
        # traffic purely because it is the only one at full resolution, and this
        # workload is bandwidth-bound (fp16 and channels_last_3d both measured
        # zero speedup, so the cost is the number of elements, not their size).
        #
        # The refine branch deliberately stays at FULL resolution: it exists
        # because the trunk's pooling destroys penumbra and interface detail,
        # so downsampling it too would remove the one path that carries high
        # frequencies. That is also why it is the more expensive half --
        # measured 92.5 ms against the trunk's 77.9 ms at hidden=16.
        # ANISOTROPIC. BEV is [B,C,D,H,W] with D the BEAM DEPTH, and the axes
        # do not carry the same detail: the lateral penumbra is ~5 mm (2.5
        # voxels at 2 mm) while depth is build-up plus a smooth exponential.
        # Isotropic stem_stride=2 throws away 8x the voxels INCLUDING lateral,
        # which is why it failed on accuracy -- it removed the penumbra the
        # refine branch exists to preserve. (2,1,1) halves the trunk's voxels
        # by pooling only the smooth axis. Accepts an int (isotropic, the old
        # behaviour) or a (D,H,W) triple.
        if isinstance(stem_stride, (list, tuple)):
            self.stem_stride = tuple(int(v) for v in stem_stride)
            if len(self.stem_stride) != 3:
                raise ValueError(f"stem_stride triple must be (D,H,W), got {stem_stride}")
        else:
            self.stem_stride = (int(stem_stride),) * 3
        if any(v < 1 for v in self.stem_stride):
            raise ValueError(f"stem_stride entries must be >= 1, got {stem_stride}")
        self._stem_pools = any(v > 1 for v in self.stem_stride)

        # Deep supervision. One zero-init 1x1x1 head per COARSE decoder level,
        # emitting the same channels as the main heads combined (gain and/or
        # residual) so an aux prediction goes through identical correction
        # algebra rather than a reduced stand-in.
        #
        # No head on the finest level: it is the map the main head already
        # consumes, so a head there re-supervises the main prediction at full
        # resolution -- the most expensive aux volume and the least
        # informative. The v2 trunk and the proton model both skip it.
        self.deep_supervision = bool(deep_supervision)
        self.n_head_channels = 1 + int(bool(use_gain))
        widths = [base_channels * (2 ** i) for i in range(depth + 1)]
        if self.deep_supervision and depth >= 2:
            # Coarsest-first, matching _UNet3D's pyramid order: levels with
            # downsample factor 2**(depth-1) ... 2**1.
            self.aux_heads = nn.ModuleList([
                self._zero_head(widths[depth - 1 - j], self.n_head_channels)
                for j in range(depth - 1)
            ])
        else:
            self.aux_heads = None

        # Additive residual head — initialised to zero so the model starts as
        # an identity (matches the original behaviour).
        self.residual_head = nn.Conv3d(b_out, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        if self.residual_head.bias is not None:
            nn.init.zeros_(self.residual_head.bias)

        if use_gain:
            self.gain_head = nn.Conv3d(b_out, 1, kernel_size=1)
            nn.init.zeros_(self.gain_head.weight)
            if self.gain_head.bias is not None:
                nn.init.zeros_(self.gain_head.bias)
        else:
            self.gain_head = None

    @staticmethod
    def _zero_head(in_channels: int, out_channels: int = 1) -> nn.Conv3d:
        """1x1x1 head, zero-initialised so the model starts as the identity."""
        head = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
        return head

    def _apply_correction(self, dose, gain_logits, res_logits):
        """The shared output algebra: logits -> CORRECTED DOSE.

        ``D_hat = relu( D * (1 + a*tanh(g)) + alpha * s * tanh(r) )``

        The main head returns the CORRECTION (the engine adds the baseline back
        in physical units); the aux heads return the corrected DOSE, because the
        deep-supervision term in losses_v2 compares them against a pooled BEV
        TARGET. Routing both through this one function is what keeps the aux
        heads optimising the same quantity as the main head instead of a
        plausible-looking different one.
        """
        residual = res_logits
        if self.bounded_residual:
            scale = dose.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-12)
            residual = self.additive_scale_frac * scale * torch.tanh(residual)
        if gain_logits is not None:
            gain = self.max_gain * torch.tanh(gain_logits)
            out = dose * (1.0 + gain) + residual
        else:
            out = dose + residual
        return F.relu(out) if self.bounded_residual else out

    @property
    def wants_aperture_mask(self) -> bool:
        """True when this model's norms need the aperture; the engine asks."""
        return self._norm_ctx is not None

    def forward(self, x, material_id=None, aperture_mask=None):
        # Channel 0 is the dose to correct.
        dose = x[:, 0:1]

        # Park the aperture for the masked norms. It is set and cleared around
        # the whole forward -- including the refine branch and the aux heads --
        # so every norm in this model sees the same field, and nothing leaks
        # into the next call if this one raises.
        if self._norm_ctx is not None:
            if aperture_mask is None:
                raise ValueError(
                    "norm='group_masked' but aperture_mask is None. The "
                    "engine caches the projected aperture as "
                    "CorrectedDoseEngine._bev_fluence and must pass the mask "
                    "through. Falling back to whole-box statistics would put "
                    "the field-size dependence straight back in, silently.")
            m = aperture_mask
            if m.dim() == 4:
                m = m.unsqueeze(1)
            self._norm_ctx.set(m.to(device=x.device, dtype=x.dtype))
        try:
            return self._forward(x, dose, material_id)
        finally:
            if self._norm_ctx is not None:
                self._norm_ctx.clear()

    def _forward(self, x, dose, material_id=None):

        if self.material_embedding is not None:
            if material_id is None:
                raise ValueError(
                    "material_embedding_dim > 0 but material_id is None. "
                    "build_bev_features returns it; the engine must pass it "
                    "through. Defaulting it to zeros would collapse all 86 "
                    "families to one and train on a constant.")
            mid = material_id
            if mid.dim() == 5:
                mid = mid.squeeze(1)
            mid = mid.to(device=x.device).long().clamp_(0, self.n_materials - 1)
            emb = self.material_embedding(mid).permute(0, 4, 1, 2, 3).to(x.dtype)
            x = torch.cat([x, emb], dim=1)

        # Run the POOLING TRUNK at 1/stem_stride resolution. The downsample was
        # missing entirely -- only the upsample-back below existed, referencing
        # an x_full that was never bound -- so stem_stride > 1 raised NameError
        # for as long as it has existed. The refine branch stays at full
        # resolution, which is the whole point of the split: cheap global
        # context from the trunk, detail from the refiner.
        x_full = x
        x_trunk = (F.avg_pool3d(x, kernel_size=self.stem_stride,
                                stride=self.stem_stride, ceil_mode=True)
                   if self._stem_pools else x)
        x_padded, (od, oh, ow) = pad_to_multiple(x_trunk,
                                                 multiple=self.trunk.pad_multiple)
        want_pyramid = self.aux_heads is not None and self.training
        if want_pyramid:
            features, pyramid = self.trunk(x_padded, return_pyramid=True)
        else:
            features, pyramid = self.trunk(x_padded), []
        features = features[..., :od, :oh, :ow]

        residual = self.residual_head(features)
        gain_lr = (self.max_gain * torch.tanh(self.gain_head(features))
                   if self.gain_head is not None else None)
        if self._stem_pools:
            # Back to full resolution BEFORE the correction algebra: `dose` is
            # full-res and the gain multiplies it pointwise.
            _sz = x_full.shape[-3:]
            residual = F.interpolate(residual, size=_sz, mode="trilinear",
                                     align_corners=False)
            if gain_lr is not None:
                gain_lr = F.interpolate(gain_lr, size=_sz, mode="trilinear",
                                        align_corners=False)
        if self.bounded_residual:
            scale = dose.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-12)
            residual = self.additive_scale_frac * scale * torch.tanh(residual)
        if gain_lr is not None:
            correction = dose * gain_lr + residual
        else:
            correction = residual
        if self.refine is not None:
            # x here is the pre-pad input; the trunk path padded/cropped
            # internally, so use the original-resolution x for the
            # full-res refinement and add its (zero-init) contribution.
            if self.refine_mode == "sequential":
                # Peak-normalise the trunk's correction so the extra channel is
                # on the same scale as channel 0, which is peak-normalised dose.
                _pk = dose.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-12)
                r = self.refine(torch.cat([x_full, correction / _pk], dim=1))
            else:
                r = self.refine(x_full)
            if self.refine_scale_to_peak:
                # PURE scale equivariance -- multiply by this control point's
                # peak and nothing else. Deliberately NOT the trunk residual's
                # `additive_scale_frac * scale * tanh(.)`, which would cap this
                # branch at 5% of peak.
                #
                # That cap would cripple the model, because this branch is not a
                # refinement in any meaningful sense -- it carries the BULK of
                # the correction. Ablated on the e18 checkpoint over 6 patients
                # x 3 beams x 180 CPs, dropping it costs 41.38 gamma points
                # (96.64 -> 55.26) and takes Level-1 beam MAE from 0.00885 to
                # 0.02769. The trunk's own additive term is bounded at 5% of
                # peak, so the architecture FORCES this branch to carry
                # whatever the gain head cannot.
                #
                # Multiplying by the peak leaves its range untouched and only
                # ties the magnitude to the control point, which is the actual
                # defect: the branch's first norm destroys the dose scale, so
                # without this it emits an absolute correction learned from the
                # training peak distribution. Smaller apertures mean lower peak
                # dose per CP -- exactly where the test set is suspected to
                # differ. Zero-init on refine.out still makes it a no-op at
                # step 0, and relu(dose + correction) still bounds it below.
                scale = dose.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-12)
                r = scale * r
            correction = correction + r
        if self.bounded_residual:
            # The engine adds this correction to `dose`, so enforcing relu on
            # the SUM is the same non-negativity v2 gets from its own relu --
            # and v1 currently emits slightly negative dose without it
            # (leak_mass reads -0.0000 on thorax).
            correction = F.relu(dose + correction) - dose

        deep: tuple = ()
        if want_pyramid and pyramid:
            aux = []
            depth = self.trunk.depth
            # zip stops at len(aux_heads) == depth - 1, which drops the finest
            # pyramid entry -- deliberately, see __init__.
            for j, (head, feat) in enumerate(zip(self.aux_heads, pyramid)):
                # Level j is downsampled by this factor from the PADDED input,
                # so crop it to the padded-away original extent (ceil division)
                # before the target is pooled onto it. Without this the aux
                # grid covers the padding too and the pooled target lands
                # spatially offset from the prediction.
                f = 2 ** (depth - 1 - j)
                dd, dh, dw = -(-od // f), -(-oh // f), -(-ow // f)
                logits = head(feat[..., :dd, :dh, :dw])
                dose_lr = F.adaptive_avg_pool3d(dose, (dd, dh, dw))
                if self.gain_head is not None:
                    g_lr, r_lr = logits[:, 0:1], logits[:, 1:2]
                else:
                    g_lr, r_lr = None, logits[:, 0:1]
                aux.append(self._apply_correction(dose_lr, g_lr, r_lr))
            # losses_v2 wants FINEST-first, so its 0.5**i discounts the coarser
            # scales rather than the finer ones.
            deep = tuple(reversed(aux))

        if self.deep_supervision:
            return {"correction": correction, "deep_supervision": deep}
        return correction


class FluenceCorrectionModel(nn.Module):
    """Tiny 2D U-Net that pre-distorts the fluence MAP before projection.

    Corrects in 2D, upstream of the physics, instead of in 3D downstream of it.
    The map is a single [H, W] plane per control point where the dose volume is
    [D, H, W], so the same receptive field costs orders of magnitude less --
    which is the whole reason to do it here: inference has to be cheap.

    Deliberately small. `depth` is the number of downsamplings and `base_channels`
    the width of the first level; the shipped 64->1024 / depth-4 version was
    28,953,409 parameters, which is 20x the 3D dose corrector it was meant to be
    cheaper than.

    The final conv is zero-initialised so the correction starts at exactly 0 and
    the engine begins from the uncorrected fluence. The engine applies the output
    ADDITIVELY, as `fluence + tanh(.)`, so the network CAN open up fluence the
    MLC model says is blocked -- which is the point if the fluence generation is
    what is wrong. The map's own maximum is 1.0, so tanh already bounds the
    correction to the map's own scale; nothing further is tuned here.

    Whether that freedom is abused is a property of the LOSS, not of this
    module: the v2 support term scores only voxels where the target is
    non-zero, so out-of-field dose costs nothing and a correction covering ~99x
    the open aperture will drift into it. Train with a loss that covers the
    whole volume (--v2_support_mode all, or --loss l1) and the leak shrinks on
    its own.
    """

    def __init__(self, base_channels: int = 8, depth: int = 2,
                 out_scale: float = 1.0):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        chs = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.encs = nn.ModuleList()
        c_in = 1
        for c in chs[:-1]:
            self.encs.append(DoubleConv(c_in, c))
            c_in = c
        self.bottleneck = DoubleConv(c_in, chs[-1])
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(chs[i + 1], chs[i], kernel_size=1)))
            self.decs.append(DoubleConv(chs[i] * 2, chs[i]))
        self.final = nn.Conv2d(chs[0], 1, kernel_size=1)
        nn.init.zeros_(self.final.weight)
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)
        self.pool = nn.MaxPool2d(2)
        self.out_scale = float(out_scale)
        self.depth = int(depth)

    def forward(self, x):
        skips = []
        h = x
        for enc in self.encs:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                # A fluence map is whatever size the MLC grid gives; it is not
                # padded to a multiple of 2**depth, so an odd axis loses a row
                # to pooling and the skip no longer lines up.
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))
        return self.out_scale * torch.tanh(self.final(h))

class PerSampleBevCrop:
    """One box per control point, all the same size, each centred on its own
    aperture -- so they still stack into one tensor.

    The shared box `compute_bev_crop` returns is the UNION over the batch, and
    SameBeamBatchSampler shuffles within a beam, so a batch is several
    unrelated apertures from across the arc. Every field in that union sits
    off-centre by however far the OTHER fields reached, and lat_h/lat_w are
    measured from the box centre -- so the same control point gets different
    coordinates depending on what it was batched with. Serving runs one CP per
    box (GC_BATCH_CHUNK=1), where the field is always centred, so training and
    inference disagree by a batch-dependent amount.

    Here each sample keeps its own H/W window, centred on its own field, and
    every window is grown to one common size so they can be stacked. Growing a
    window takes the real neighbouring voxels; only a window running off the
    volume edge sees zeros, exactly as the shared box already did. Depth stays
    shared: it is bounded by the DOSE extent rather than the aperture, and the
    depth channel is absolute (measured from the source), so re-centring it
    would change what it means.
    """

    __slots__ = ("d", "h_starts", "w_starts", "size_h", "size_w")

    def __init__(self, d_slice, h_starts, w_starts, size_h, size_w):
        if len(h_starts) != len(w_starts):
            # zip() would silently truncate to the shorter one, i.e. drop a
            # control point's dose on the floor with no error anywhere.
            raise ValueError(
                f"h_starts and w_starts disagree on the batch size: "
                f"{len(h_starts)} vs {len(w_starts)}")
        self.d = d_slice
        self.h_starts = list(int(x) for x in h_starts)
        self.w_starts = list(int(x) for x in w_starts)
        self.size_h = int(size_h)
        self.size_w = int(size_w)

    def __len__(self):
        return len(self.h_starts)

    def sample_slices(self, i):
        h, w = self.h_starts[i], self.w_starts[i]
        return (self.d, slice(h, h + self.size_h), slice(w, w + self.size_w))

    @property
    def shape(self):
        return (self.d.stop - self.d.start, self.size_h, self.size_w)

    def apply(self, x):
        """[N, C, D, H, W] -> [N, C, d, size_h, size_w]."""
        if x is None:
            return None
        sd = self.d
        return torch.stack([
            x[i][:, sd, h:h + self.size_h, w:w + self.size_w]
            for i, (h, w) in enumerate(zip(self.h_starts, self.w_starts))
        ], dim=0)

    def axes(self, lat_h_mm, lat_w_mm):
        """Per-sample lateral axis values, [N, size_h] and [N, size_w].

        These are what make the coordinate channels agree across the batch:
        each sample's axis is centred on its own field, so once
        build_bev_features subtracts the per-sample midpoint every sample sees
        the same relative coordinates.
        """
        h = torch.stack([lat_h_mm[s:s + self.size_h] for s in self.h_starts])
        w = torch.stack([lat_w_mm[s:s + self.size_w] for s in self.w_starts])
        return h, w

    def scatter(self, out_flat, full_shape):
        """Write per-sample boxes back into a zero volume of `full_shape`."""
        full = out_flat.new_zeros(full_shape)
        sd = self.d
        for i, (h, w) in enumerate(zip(self.h_starts, self.w_starts)):
            full[i][:, sd, h:h + self.size_h, w:w + self.size_w] = out_flat[i]
        return full


def _fluence_weighted_tile_depth(src_tiles, rad_depth, density, fallback,
                                 body_density_threshold=0.02):
    """Representative ``[N,T,D]`` depth without selecting one spatial ray.

    ``src_tiles`` is ``[N,T,D,H,W]`` and the other volumes broadcast over T.
    Air is excluded explicitly because radiological depth remains positive
    after a ray exits the patient. Empty tile/depth planes retain the sampled
    fallback ray so closed fields and downstream air stay well-defined.
    """
    inside = (density > body_density_threshold).to(src_tiles.dtype)
    weights = src_tiles * inside
    weight_sum = weights.sum(dim=(-2, -1))
    mean_depth = (weights * rad_depth).sum(dim=(-2, -1)) / weight_sum.clamp_min(1e-12)
    return torch.where(weight_sum > 1e-12, mean_depth, fallback)


class CorrectedDoseEngine(nn.Module):
    """
    Implements the full dose calculation pipeline for radiotherapy.

    Usage:
        engine = DoseEngine(machine_config)
        dose = engine.forward(leaf_positions, mus, jaw_positions, density_image)

    Or with BeamSequence:
        dose = engine.forward_beam_sequence(beam_seq, density_image)

    Attributes:
        machine_config (MachineConfig): Machine physics parameters.
        device (torch.device): PyTorch device for computation.
    """
    machine_config: MachineConfig | None = None
    dose_grid_shape: tuple[int, int, int] | None = None
    dose_grid_spacing: tuple[float, float, float] | None = None
    number_of_beams: int | None = None
    layers_initialized: bool = False
    gantry_angles: torch.Tensor | None = None
    collimator_angles: torch.Tensor | None = None
    field_size: tuple[float, float] | None = None
    SID: float | None = None
    iso_center: tuple[float, float, float] | None = None

    def __init__(
        self,
        machine_config: MachineConfig,
        kernel_size: int,
        dose_grid_spacing: tuple[float, float, float],
        dose_grid_shape: tuple[int, int, int],
        beam_template: BeamSequence | Beam | None = None,        
        auto_calibrate: bool = False,
        adjust_values: bool = None,
        fluence_correction_model: nn.Module = None,
        fluence_correction_grad: bool = False,
        dose_correction_model: nn.Module = None,
        fluence_to_dose: bool = False,
        multislab: bool = False,
        mu_eff: float = 0.05,
        lateral_scatter: bool = False,
        lateral_scatter_model: nn.Module = None,
        lateral_scatter_grad: bool = False,
        lat_sigma_mm: float = 1.5,
        lat_cap_mm: float = None,
        lat_fine: bool = True,
        lattice_size: int = 1,
        lattice_tile_chunk: int = 4,
        lattice_depth_mode: str = "ray",
        terma_scaling: bool = False,
        terma_c1: float | None = None,
        terma_c2_per_mm: float | None = None,
        terma_smoothing_mm: float = 10.0,
        terma_detach_field_size: bool = False,
        terma_learnable: bool = False,
        allow_terma_with_lateral_scatter: bool = False,
        use_source_distance: bool = False,
        # Feed the v2 BEV feature stack to a NON-v2 corrector. Kept separate
        # from the model type on purpose: the feature set and the trunk are two
        # different questions and were not separable while feature-building was
        # inlined in the v2-only forward path.
        v2_features: bool = False,
        # Autocast dtype for the CORRECTOR forward only, or None for fp32.
        # Scoped the way the proton path scopes it: around the network, never
        # around the physics. The dose engine accumulates small numbers
        # (per-CP peaks are ~7e-5 before the x1e5 gain) and has no business in
        # reduced precision; the corrector is where the activations live and
        # where the memory goes. bf16 keeps fp32 exponent range, so unlike
        # fp16 it needs no GradScaler.
        amp_dtype=None,
        # The SAME FeatureConfig the corrector was sized from. Must be passed:
        # build_bev_features defaults to FeatureConfig() otherwise, which emits
        # the "film" channel count while the model was built for whatever
        # --v2_cond_mode asked for, and the stem gets 21 channels where it
        # wants 24.
        v2_feature_cfg=None,
        # Crop the BEV volume to the region the beam actually irradiates,
        # BETWEEN the rotation and the corrector.
        #
        # loaders.crop_lateral_to_body records why this cannot be done in
        # patient space: pad_and_crop_to_iso_center PADS D and W so the
        # isocentre lands at the geometric centre, which can nearly double a
        # lateral axis, and the corrector pays for every air voxel -- but
        # cropping there to the body's axis-aligned extent silently destroys
        # the dose at oblique gantry angles (measured, ~100% of peak), because
        # the BEV rotation needs the box to contain the ROTATED body. The
        # rotation-safe version crops to the body RADIUS and recovers 1.03x,
        # i.e. nothing. Hence: "the saving has to be taken in BEV space
        # instead, where the beam is axis-aligned and the crop can follow the
        # field". That is here.
        #
        # The invariant this must preserve: every feature value at a voxel
        # INSIDE the crop is bit-identical to its value in the uncropped run.
        # Pinned by tests/test_bev_crop.py. It is what makes crop-on/crop-off
        # an honest A/B on the same weights, and it is not free -- the
        # cumulative electron path is a cumsum along D, so it has to be
        # computed on the FULL volume and sliced, never computed on the slice.
        bev_crop: bool = False,
        # Margin in mm around the irradiated support. This is the accuracy
        # knob: too tight and the corrector loses the scatter halo it exists to
        # model, and the additive head cannot build a penumbra tail it cannot
        # see. Lateral scatter sigma reaches 8 mm in lung with a 25 mm cap
        # (configs.MULTISLAB_DEFAULTS), so 50 mm is ~2x the cap and
        # deliberately generous -- tighten it only against a measurement.
        bev_crop_margin_mm: float = 50.0,
        # Floor on each cropped axis, in voxels. A closed-MLC control point has
        # no support at all and would otherwise crop to nothing.
        bev_crop_min: int = 32,
        # One box per control point instead of one union box per batch. See
        # PerSampleBevCrop: the union makes the lateral coordinate channels
        # depend on which CPs were drawn together, and serving never draws
        # more than one. Opt-in because it changes what a trained corrector
        # was fitted to; recorded in model_config like the rest.
        bev_crop_per_sample: bool = False,
        # Apply the field crop before the pencil-beam convolution and
        # heterogeneity physics as well as before the corrector.  The work box
        # is derived from projected fluence (available before convolution) and
        # automatically enlarged by the finite PB/scatter support not already
        # covered by bev_crop_margin_mm.  Opt-in until full-plan parity and
        # runtime are established on the deployment GPU.
        early_bev_crop: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = None,
        verbose: bool = False,
    ) -> "CorrectedDoseEngine":
        """
        Initializes the CorrectedDoseEngine pipeline.

        Args:
            machine_config: Machine physics and MLC specifications.
            kernel_size: Size of the pencil beam dose kernel.
            dose_grid_spacing: Voxel spacing in mm (depth, height, width).
            dose_grid_shape: Shape of the output grid tensor (depth, height, width) in pixels.            
            beam_template: Optional Beam or BeamSequence defining the treatment geometry.
                If omitted, the engine is unconfigured until the first compute_dose call.
            auto_calibrate: Run calibration immediately after construction (default: False).
            adjust_values: Deprecated adjustment of beam parameters.
            device: PyTorch device for computation.
            dtype: Data type for tensors.
            verbose: Enable verbose output (default: False).
        """
        super().__init__()
        self.kernel_size = kernel_size

        # Handle device default
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        self.machine_config = machine_config
        self.dose_grid_spacing = dose_grid_spacing
        self.dose_grid_shape = dose_grid_shape
        # When True: skip the pencil-beam kernel convolution AND ALL fluence
        # modelling (penumbra, head scatter, output factor, profile, MLC
        # transmission). The CNN is fed a HARD-APERTURE fluence map projected to
        # 3D (a plain geometric projection, no convolutions), and its output IS
        # the dose — so the CNN learns the entire kernel. Set before
        # _initialize_layers so the fluence-map layer is built pure-aperture.
        self.fluence_to_dose = fluence_to_dose
        if fluence_to_dose:
            self._fluence_map_config = machine_config.model_copy(update={
                "penumbra_fwhm": None,
                "head_scatter_amplitude": None,
                "head_scatter_sigma": None,
                "output_factors": None,
                "profile_corrections": None,
                "mlc_transmission": 0.0,
            })
        else:
            self._fluence_map_config = machine_config
        self._initialize_layers(beam_template)
        self.fluence_correction_model = fluence_correction_model
        # The historical fluence-correction hook was inference-only: its
        # forward lived under no_grad and train_correction.py never put its
        # parameters in the optimizer.  Keep that behaviour by default so an
        # old caller cannot suddenly retain the entire differentiable physics
        # graph.  Focused fluence-only experiments opt in explicitly.
        self.fluence_correction_grad = bool(fluence_correction_grad)
        self.dose_correction_model = dose_correction_model
        # Multislab: apply a per-voxel radiological-depth heterogeneity correction to
        # the pencil-beam baseline, and feed that radiological depth to the corrector
        # as an extra input channel. mu_eff is the correction strength (see MultislabEngine).
        self.multislab = multislab
        self.mu_eff = mu_eff
        self.lateral_scatter = lateral_scatter
        self.lateral_scatter_model = lateral_scatter_model
        self.lateral_scatter_grad = bool(lateral_scatter_grad)
        if self.lateral_scatter_model is not None and not multislab:
            raise ValueError("learned lateral scatter requires multislab=True")
        self.lat_sigma_mm = lat_sigma_mm
        self.lat_cap_mm = lat_cap_mm
        self.lat_fine = lat_fine
        # lattice_size = 1 keeps the shipped single-central-ray path bit for
        # bit; > 1 splits the aperture into lattice_size^2 equal-fluence tiles,
        # each with its own divergent ray, kernel and multislab residual.
        self.lattice_size = int(lattice_size)
        self.lattice_tile_chunk = int(lattice_tile_chunk)
        # "bev"     : the corrector runs in beam's-eye-view and its output is
        #             rotated to the patient frame afterwards (the shipped path).
        # "patient" : the BEV dose AND the whole feature stack are rotated to
        #             the patient frame first, and the corrector runs there.
        #
        # BEV aligns the beam axis with an array axis, so build-up, attenuation
        # and range all run along one consistent dimension and a 3D conv sees
        # the same physics orientation at every control point -- orientation
        # invariance for free. Patient space gives that up and must learn it,
        # which is why the published patient-space networks (Tsekas 2022,
        # Tseng 2023) randomise gantry angles over 0-359 degrees and run
        # 5-90M parameters. What it buys: no resampling of the PREDICTION, and
        # the correction is learned on the grid the metric is computed on.
        if lattice_depth_mode not in ("ray", "tile_mean"):
            raise ValueError("lattice_depth_mode must be 'ray' or 'tile_mean', "
                             f"got {lattice_depth_mode!r}")
        self.lattice_depth_mode = lattice_depth_mode
        if self.lattice_size < 1:
            raise ValueError(f"lattice_size must be >= 1, got {lattice_size}")
        if self.lattice_size >= 1 and not multislab:
            raise ValueError(
                "the lattice requires multislab=True: the lattice IS the "
                "multislab residual, evaluated per tile instead of against one "
                "ray through the isocentre. Without it the per-tile residual "
                "would be computed and then never applied.")
        self.terma_scaling = bool(terma_scaling)
        if self.terma_scaling:
            if not multislab:
                raise ValueError("TERMA scaling requires multislab=True")
            if lateral_scatter and not allow_terma_with_lateral_scatter:
                raise ValueError(
                    "TERMA scaling and heuristic lateral_scatter both target "
                    "lateral disequilibrium; evaluate them separately or set "
                    "allow_terma_with_lateral_scatter=True for an explicit ablation")
            if terma_c1 is None or terma_c2_per_mm is None:
                raise ValueError(
                    "TERMA scaling requires explicit terma_c1 and terma_c2_per_mm")
            self.terma_scaling_layer = TermaScalingLayer(
                c1=terma_c1,
                c2_per_mm=terma_c2_per_mm,
                spacing_mm=dose_grid_spacing,
                smoothing_size_mm=terma_smoothing_mm,
                detach_field_size=terma_detach_field_size,
                learnable=terma_learnable,
            )
        else:
            self.terma_scaling_layer = None
        self.terma_scale = None
        self.terma_field_size_mm = None
        self.use_source_distance = use_source_distance
        self.v2_features = bool(v2_features)
        self.amp_dtype = amp_dtype
        self.v2_feature_cfg = v2_feature_cfg
        self.bev_crop = bool(bev_crop)
        self.bev_crop_margin_mm = float(bev_crop_margin_mm)
        self.bev_crop_min = int(bev_crop_min)
        self.bev_crop_per_sample = bool(bev_crop_per_sample)
        self.early_bev_crop = bool(early_bev_crop)
        self.early_bev_crop_slices = None
        # Set by every corrector forward: the slices actually used, or None.
        # The training loop needs them to crop the deep-supervision target the
        # same way -- the aux predictions live on the CROPPED grid, and pooling
        # a full-size target onto them would land spatially offset.
        self.bev_crop_slices = None
        if adjust_values is not None:
            raise ValueError("The `adjust_values` argument, together with the beam validation layer has been removed due to major limitations.")
        
        if auto_calibrate:
            self.calibrate(verbose=verbose)
            self._initialize_layers(beam_template)    

    def _set_device_dtype(self, device, dtype) -> None:
        if self.dtype is None:
            self.dtype = dtype
        if self.device is None:
            self.device = device

    def _initialize_layers(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        if new_beam_data is None:
            return

        initialize_fluence_map_layer = not hasattr(self, 'fluence_map_layer')
        initialize_fluence_volume_layer = not hasattr(self, 'fluence_volume_layer')
        initialize_beam_wise_conv_layer = not hasattr(self, 'beam_wise_conv_layer')
        initialize_pencil_beam_kernel_layer = not hasattr(self, 'pencil_beam_kernel_layer')
        initialize_rad_depth_layer = not hasattr(self, 'rad_depth_layer')
        initialize_rotation_layer = not hasattr(self, 'rotation_layer')

        if isinstance(new_beam_data, Beam):
            number_of_beams = 1
            gantry_angles = torch.tensor([new_beam_data.gantry_angle]).to(self.dtype).to(self.device)
            collimator_angles = torch.tensor([new_beam_data.collimator_angle]).to(self.dtype).to(self.device)
        elif isinstance(new_beam_data, BeamSequence):
            number_of_beams = len(new_beam_data)
            gantry_angles = new_beam_data.gantry_angles
            collimator_angles = new_beam_data.collimator_angles.to(self.dtype).to(self.device)

        if self.dtype is None:
            self.dtype = new_beam_data.dtype
        if self.device is None:
            self.device = new_beam_data.device
        
        if  self.number_of_beams is None or (self.number_of_beams != number_of_beams):
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        elif self.gantry_angles is None or (self.gantry_angles != gantry_angles).any():
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        elif self.collimator_angles is None or (self.collimator_angles != collimator_angles).any():
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        self.number_of_beams = number_of_beams
        self.gantry_angles = gantry_angles
        self.collimator_angles = collimator_angles
        # Precompute "is any collimator angle non-zero" as a Python
        # bool at init time. forward() used to do this check per call,
        # which forces a CUDA->host sync every step — invisible at
        # G=1 but a real GPU stall at G>1.
        self._has_collimator_rotation = bool((collimator_angles != 0.0).any().item())


        if self.field_size is None or (self.field_size != new_beam_data.field_size):
            initialize_fluence_map_layer = True
            initialize_fluence_volume_layer = True
        self.field_size = new_beam_data.field_size


        self.SID = new_beam_data.sid
        if self.iso_center is None or (self.iso_center != new_beam_data.iso_center):
            initialize_fluence_volume_layer = True
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        self.iso_center = new_beam_data.iso_center

        if self.dtype is None:
            return
        if self.device is None:
            return
        if self.dose_grid_shape is None:
            return
        if self.dose_grid_spacing is None:
            return
        if self.number_of_beams is None:
            return
        
        if initialize_fluence_map_layer:
            self.fluence_map_layer = FluenceMapLayer(
                getattr(self, "_fluence_map_config", self.machine_config),
                device = self.device,
                dtype=self.dtype,
                field_size=self.field_size,
                verbose=self.verbose
            )
        
        if initialize_fluence_volume_layer:
            self.fluence_volume_layer = FluenceVolumeLayer(
                self.machine_config, 
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                sid=self.SID,
                iso_center=self.iso_center,
                field_size=self.field_size,
                verbose=self.verbose
            )

        if initialize_rad_depth_layer:
            self.rad_depth_layer = RadiologicalDepthLayer(
                self.machine_config, 
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,
                verbose=self.verbose
            )

        if initialize_pencil_beam_kernel_layer:
            self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
                self.machine_config, 
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                kernel_size=self.kernel_size,
                verbose=self.verbose
            )

        if initialize_beam_wise_conv_layer:
            self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(
                self.device, 
                self.dtype,
                verbose=self.verbose
            )

        if initialize_rotation_layer:
            self.rotation_layer = BeamRotationLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,
                resolution=self.dose_grid_spacing,
                verbose=self.verbose
            )
            # Inverse (patient -> BEV) rotation. Used internally for the
            # rotated density image fed into the corrector, and exposed
            # so the trainer can rotate target dose into BEV for IDD
            # curve computation.
            self.inv_rotation_layer = BeamRotationLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=-self.gantry_angles,
                iso_center=self.iso_center,
                resolution=self.dose_grid_spacing,
                verbose=self.verbose
            )

        self.layers_initialized = True

    # ------------------------------------------------------------------
    # Multilattice pencil beam
    # ------------------------------------------------------------------
    def _lattice_tiles(self, fluence_flat, crop):
        """Equal-fluence tiles, per sample.

        Returns ``(ray_depth [N, T, D], tile_id [N, D, H, W], T)`` where every
        sample gets its OWN tile edges -- apertures differ at every control
        point, so a shared partition would put one CP's field in another's
        tile. ``T`` is ``lattice_size**2`` for all samples (empty tiles are
        kept, contributing zero) so the tile axis stays rectangular and the
        whole batch can go through one grouped convolution.

        ``crop`` is the ``early_bev_crop`` slice triple or None. The tile and
        ray geometry is defined by the isocentre in FULL BEV indices, so the
        crop offset has to be subtracted or every ray is displaced by the crop
        origin -- which is silent, because the result is still a plausible dose.
        """
        n = int(self.lattice_size)
        N, d, h, w = fluence_flat.shape[0], *fluence_flat.shape[-3:]
        dev, dt = fluence_flat.device, fluence_flat.dtype
        r_h, r_d, r_w = self.dose_grid_spacing
        off_d = off_h = off_w = 0
        if crop is not None:
            off_d, off_h, off_w = crop[0].start, crop[1].start, crop[2].start
        iso_h = float(self.iso_center[0]) / r_h - off_h
        iso_d = float(self.iso_center[1]) / r_d - off_d
        iso_w = float(self.iso_center[2]) / r_w - off_w

        # The tile partition is defined on the plane through the isocentre, so
        # tile bounds are isocentre-plane coordinates and back-projection is a
        # pure scale -- the same convention the ray sampling below uses.
        idx = int(min(max(iso_d, 0), d - 1))
        plane = fluence_flat[:, 0, idx].clamp_min(0.0)            # [N, h, w]
        edges_h = self._equal_fluence_edges(plane.sum(-1), n, h)  # [N, n+1]
        edges_w = self._equal_fluence_edges(plane.sum(-2), n, w)  # [N, n+1]

        hg = torch.arange(h, device=dev, dtype=dt).view(1, h, 1)
        wg = torch.arange(w, device=dev, dtype=dt).view(1, 1, w)
        ch = torch.zeros(N, n * n, device=dev, dtype=dt)
        cw = torch.zeros(N, n * n, device=dev, dtype=dt)
        for i in range(n):
            for j in range(n):
                m = ((hg >= edges_h[:, i].view(N, 1, 1))
                     & (hg < edges_h[:, i + 1].view(N, 1, 1))
                     & (wg >= edges_w[:, j].view(N, 1, 1))
                     & (wg < edges_w[:, j + 1].view(N, 1, 1)))
                wgt = (plane * m).sum(dim=(-2, -1))
                safe = wgt.clamp_min(1e-12)
                t = i * n + j
                # An empty tile has no centroid; fall back to the isocentre so
                # the ray is well defined. It contributes zero dose regardless,
                # because its source is zero everywhere.
                ch[:, t] = torch.where(wgt > 0,
                                       (plane * m * hg).sum(dim=(-2, -1)) / safe,
                                       torch.full_like(wgt, iso_h))
                cw[:, t] = torch.where(wgt > 0,
                                       (plane * m * wg).sum(dim=(-2, -1)) / safe,
                                       torch.full_like(wgt, iso_w))

        # Voxel -> tile, by back-projecting each voxel to the isocentre plane.
        z = torch.arange(d, device=dev, dtype=dt)
        scale = ((float(self.SID) + (z - iso_d) * r_d)
                 / float(self.SID)).clamp_min(1e-3)               # [d]
        h_at = iso_h + (torch.arange(h, device=dev, dtype=dt).view(1, h)
                        - iso_h) / scale.view(d, 1)               # [d, h]
        w_at = iso_w + (torch.arange(w, device=dev, dtype=dt).view(1, w)
                        - iso_w) / scale.view(d, 1)               # [d, w]
        hi = torch.searchsorted(
            edges_h[:, 1:-1].contiguous(),
            h_at.reshape(1, -1).expand(N, -1).contiguous()).view(N, d, h)
        wi = torch.searchsorted(
            edges_w[:, 1:-1].contiguous(),
            w_at.reshape(1, -1).expand(N, -1).contiguous()).view(N, d, w)
        tile_id = hi.unsqueeze(-1) * n + wi.unsqueeze(-2)         # [N, d, h, w]
        return ch, cw, tile_id, n * n

    @staticmethod
    def _equal_fluence_edges(profile, n, size):
        """Bin edges splitting each row of ``profile`` into n equal-weight parts."""
        N = profile.shape[0]
        c = profile.clamp_min(0.0).cumsum(-1)
        total = c[:, -1:].clamp_min(1e-12)
        q = torch.arange(1, n, device=profile.device,
                         dtype=profile.dtype).view(1, -1) / n
        # +1 because searchsorted returns the index j with cumsum[j] >= target,
        # so the first part must INCLUDE j and the exclusive upper edge is j+1.
        # Without it every tile's lower boundary sits one voxel too low, which
        # changes only the boundary planes -- 0.15% of voxels, identical MAE --
        # and so is invisible in any aggregate metric.
        inner = torch.searchsorted(c.contiguous(), (total * q).contiguous()) + 1
        inner = inner.clamp(1, size - 1).to(profile.dtype)
        lo = torch.zeros(N, 1, device=profile.device, dtype=profile.dtype)
        hi = torch.full((N, 1), float(size), device=profile.device,
                        dtype=profile.dtype)
        # cummax keeps the edges non-decreasing when a degenerate profile puts
        # two quantiles in the same voxel; without it searchsorted can emit a
        # descending pair and a tile silently spans negative width.
        return torch.cat([lo, inner.cummax(dim=1)[0], hi], dim=1)

    def _lattice_ray_depths(self, rad_depth_flat, ch, cw, crop):
        """Divergent radiological depth along every tile's ray: [N, T, D]."""
        N, _, d, h, w = rad_depth_flat.shape
        T = ch.shape[1]
        r_h, r_d, r_w = self.dose_grid_spacing
        off_d = off_h = off_w = 0
        if crop is not None:
            off_d, off_h, off_w = crop[0].start, crop[1].start, crop[2].start
        iso_h = float(self.iso_center[0]) / r_h - off_h
        iso_d = float(self.iso_center[1]) / r_d - off_d
        iso_w = float(self.iso_center[2]) / r_w - off_w
        z = torch.arange(d, device=ch.device, dtype=ch.dtype)
        scale = (float(self.SID) + (z - iso_d) * r_d) / float(self.SID)   # [d]
        ray_h = iso_h + (ch.unsqueeze(-1) - iso_h) * scale.view(1, 1, d)
        ray_w = iso_w + (cw.unsqueeze(-1) - iso_w) * scale.view(1, 1, d)
        gx = 2.0 * (ray_w + 0.5) / w - 1.0
        gy = 2.0 * (ray_h + 0.5) / h - 1.0
        gz = (2.0 * (z + 0.5) / d - 1.0).view(1, 1, d).expand_as(gx)
        grid = torch.stack((gx, gy, gz), dim=-1).view(N, T, d, 1, 3)
        return F.grid_sample(rad_depth_flat, grid, mode="bilinear",
                             padding_mode="border",
                             align_corners=False)[:, 0, :, :, 0]

    def lattice_pencil_beam(self, fluence_vol, rad_depth_flat, density_flat, crop,
                            tile_chunk=4):
        """The N x N ray pencil beam, replacing the single central-ray one.

        Each equal-fluence tile gets its own divergent density ray, hence its
        own depth-dependent kernel, and its own multislab residual relative to
        THAT ray -- which is the whole point: the shipped path corrects every
        voxel against one ray through the isocentre, so anatomy laterally
        offset from it is corrected with the wrong path.

        Tiles are batched into the convolution's leading dimension, which
        ``BeamWiseConvolutionalLayer`` already turns into a grouped conv with
        ``BG*D`` groups. Chunking bounds memory; the arithmetic is inherently
        T-fold, so this removes launch overhead, not work.

        Returns dose ``[N, D, H, W, 1]`` with the per-tile residual applied,
        so the caller must NOT apply the single-ray ``cf`` afterwards.
        """
        fl = fluence_vol.squeeze(-1)                         # [N, D, H, W]
        N, d, h, w = fl.shape
        # Tile geometry and kernels are PHYSICS, not learned, and the kernel
        # layer writes through `out=` arguments, which autograd refuses. The
        # shipped single-ray path builds its kernels under no_grad for exactly
        # this reason; without the same treatment the first training step dies
        # in pencil_beam_model.get_pencil_beam.
        with torch.no_grad():
            ch, cw, tile_id, T = self._lattice_tiles(fl.unsqueeze(1).detach(),
                                                     crop)
            ray = self._lattice_ray_depths(
                rad_depth_flat.detach(), ch, cw, crop).detach()   # [N,T,D]
        out = fl.new_zeros(N, d, h, w)
        for s in range(0, T, tile_chunk):
            k = min(tile_chunk, T - s)
            sel = torch.arange(s, s + k, device=fl.device).view(1, k, 1, 1, 1)
            src = (fl.unsqueeze(1) * (tile_id.unsqueeze(1) == sel)
                   ).reshape(N * k, d, h, w)
            reference_depth = ray[:, s:s + k]
            if self.lattice_depth_mode == "tile_mean":
                # One representative DEPTH per tile and plane, rather than one
                # representative spatial ray.  Only voxels currently inside
                # the patient contribute, so a sloped surface starts building
                # dose as soon as any irradiated part of the tile enters. This
                # avoids the zero-kernel entrance gap caused by a centroid ray
                # that is still in air, without generating any extra kernels.
                reference_depth = _fluence_weighted_tile_depth(
                    src.detach().view(N, k, d, h, w),
                    rad_depth_flat.detach().view(N, 1, d, h, w),
                    density_flat.detach().view(N, 1, d, h, w),
                    reference_depth)
            with torch.no_grad():
                kern = self.pencil_beam_kernel_layer(
                    (reference_depth * 10.0).reshape(N * k, d, 1)).detach()
            dose = self.beam_wise_conv_layer(
                src.unsqueeze(-1), kern).view(N, k, d, h, w)
            resid = torch.exp(
                -self.mu_eff * (rad_depth_flat.view(N, 1, d, h, w)
                                - reference_depth.view(N, k, d, 1, 1))
            ).clamp(0.3, 3.0)
            tile_dose = dose * resid
            out = out + tile_dose.sum(1)
            del src, kern, dose, resid, tile_dose, reference_depth
        return out.unsqueeze(-1)

    def bev_axis_coords(self):
        """(depth_mm, lat_h_mm, lat_w_mm) for the BEV grid, in mm.

        Single source of truth for the BEV axis convention, shared by
        ``source_distance_bev`` and the v2 feature builder so the two cannot
        drift. ``depth_mm`` is measured FROM THE SOURCE (so ~SAD at isocentre);
        the lateral pair is measured from the central axis.

        BEV spatial axes are (D=beam depth, H=patient z / MLC leaf axis,
        W=in-plane lateral).
        """
        H_, D_, W_ = self.dose_grid_shape
        res_h, res_d, res_w = self.dose_grid_spacing
        SAD = float(self.SID)
        d_idx = torch.arange(D_, device=self.device, dtype=torch.float32)
        h_idx = torch.arange(H_, device=self.device, dtype=torch.float32)
        w_idx = torch.arange(W_, device=self.device, dtype=torch.float32)
        depth_mm = (d_idx - self.iso_center[1] / res_d) * res_d + SAD
        lat_h_mm = (h_idx - self.iso_center[0] / res_h) * res_h
        lat_w_mm = (w_idx - self.iso_center[2] / res_w) * res_w
        return depth_mm, lat_h_mm, lat_w_mm

    def _aperture_mask_bev(self, B_, G_, Dz, Hz, Wz, crop, dtype):
        """In-field mask for the masked norms, on the corrector's own grid.

        The projected aperture the physics already built (`_bev_fluence`),
        thresholded at the same fraction of peak the conditioning area uses so
        "the aperture" means one thing everywhere. Cropped alongside the
        feature stack when --bev_crop is on.

        Raises rather than substituting ones: an all-ones mask is exactly the
        whole-box normalisation this replaces, and it would look like it was
        working.
        """
        flu = getattr(self, "_bev_fluence", None)
        if flu is None:
            raise ValueError(
                "norm='group_masked' needs the projected aperture, but the "
                "engine has no _bev_fluence. It is built by the fluence-volume "
                "layer for the pencil-beam path."
                "it.")
        flu = flu.reshape(B_ * G_, 1, Dz, Hz, Wz)
        if crop is not None:
            flu = (crop.apply(flu) if isinstance(crop, PerSampleBevCrop)
                   else flu[..., crop[0], crop[1], crop[2]])
        peak = flu.detach().amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
        thr = getattr(getattr(self, "v2_feature_cfg", None),
                      "aperture_threshold", 0.5)
        return (flu.detach() >= thr * peak).to(dtype)

    def compute_bev_crop(self, dose_flat, fluence_flat, unet_multiple=16,
                         per_sample=False):
        """Slices bounding the region the corrector actually needs.

        Each axis takes the support that actually bounds it. The FLUENCE
        support says where the beam is -- the aperture already projected along
        diverging rays -- so it bounds H and W tightly and says nothing about D
        (it spans the whole depth). The DOSE support bounds D, since dose exists
        from the body entrance to the exit, and says little about H and W, where
        scatter carries it well past the field.

        The union is taken over the whole batch: SameBeamBatchSampler hands the
        corrector several control points of ONE beam, and they go through as
        one tensor.

        `per_sample=True` returns a PerSampleBevCrop instead: H and W get one
        window per control point, centred on ITS aperture and all grown to a
        common size so they still stack. That removes the batch from the
        meaning of the lateral coordinate channels -- see PerSampleBevCrop.

        Three things the caller gets for free:
          * a margin, in mm, so the corrector keeps the scatter halo and the
            room to build a penumbra tail where the baseline has none;
          * a floor, so a closed-MLC control point (no support at all) cannot
            crop to an empty box;
          * sizes rounded UP to ``unet_multiple``, so DoseCorrectionModel's
            internal pad_to_multiple becomes a no-op and the aux-head crops in
            deep supervision come out exact.
        """
        res_h, res_d, res_w = self.dose_grid_spacing
        shape_source = dose_flat if dose_flat is not None else fluence_flat
        if shape_source is None:
            raise ValueError("compute_bev_crop needs dose_flat or fluence_flat")
        _, Dz, Hz, Wz = shape_source.shape[0], *shape_source.shape[2:]
        sizes, spacings = (Dz, Hz, Wz), (float(res_d), float(res_h), float(res_w))

        def _norm_support(vol):
            if vol is None:
                return None
            peak = vol.detach().amax(dim=(-3, -2, -1), keepdim=True)
            # An all-zero volume (closed MLC) must contribute NO support rather
            # than "everything", which is what dividing by a clamped ~0 peak
            # would give after thresholding.
            if not bool((peak > 1e-12).any()):
                return None
            return (vol.detach() / peak.clamp_min(1e-12)) > 1e-3

        # WHICH support bounds WHICH axis. The two answer different questions:
        # the projected aperture bounds H and W (it is the field, cast along
        # diverging rays) and spans the whole depth; the dose bounds D (it runs
        # from the body entrance to the exit) and spreads laterally well past
        # the field through scatter. Unioning both on all three axes -- which
        # is what this did until now -- let the dose's lateral spread widen H
        # and W, costing 32-64 mm of box in W on 1ABB045 beam 0 for nothing:
        # the fluence-only box already contains every voxel the corrector can
        # act on. Each axis therefore takes its own support, and falls back to
        # the other only when its preferred one is missing (a closed MLC has no
        # fluence support).
        flu_sup = _norm_support(fluence_flat)
        dose_sup = _norm_support(dose_flat)
        axis_support = {
            0: flu_sup if dose_sup is None else dose_sup,      # D <- dose
            1: dose_sup if flu_sup is None else flu_sup,       # H <- fluence
            2: dose_sup if flu_sup is None else flu_sup,       # W <- fluence
        }
        slices = []
        for axis, (n, spacing) in enumerate(zip(sizes, spacings)):
            supports = [s for s in (axis_support[axis],) if s is not None]
            # sup is [N, 1, D, H, W]; axis 0/1/2 -> D/H/W -> dim axis+1 once
            # the channel axis is reduced away.
            others = [a for a in (0, 1, 2) if a != axis]
            # Reduce every support's bounds on device, then read them all back
            # at once. Per support this used to be a torch.nonzero plus two
            # int() casts -- three syncs each, on an axis loop, on a forward.
            bounds = []
            for sup in supports:
                prof = sup.any(dim=1)                       # [N, D, H, W]
                for a in sorted(others, reverse=True):      # high dims first
                    prof = prof.any(dim=a + 1)
                prof = prof.any(dim=0)                      # [n], union over batch
                ar = torch.arange(prof.shape[0], device=prof.device)
                bounds.append(torch.stack((
                    torch.where(prof, ar, torch.full_like(ar, prof.shape[0])).amin(),
                    torch.where(prof, ar, torch.full_like(ar, -1)).amax())))
            lo, hi = None, None
            if bounds:
                for a_lo, a_hi in torch.stack(bounds).tolist():   # ONE sync
                    if a_lo > a_hi:                     # this support is empty
                        continue
                    a_hi += 1
                    lo = a_lo if lo is None else min(lo, a_lo)
                    hi = a_hi if hi is None else max(hi, a_hi)
            if lo is None:                                  # no support at all
                lo, hi = n // 2, n // 2 + 1
            m = int(math.ceil(self.bev_crop_margin_mm / max(spacing, 1e-6)))
            lo, hi = max(0, lo - m), min(n, hi + m)
            # Grow to the floor and then to a multiple, centred on the box, and
            # slide (never shrink) if that runs off an edge.
            want = max(self.bev_crop_min, unet_multiple)
            want = max(want, int(math.ceil((hi - lo) / unet_multiple) * unet_multiple))
            want = min(want, n)
            if hi - lo < want:
                grow = want - (hi - lo)
                lo -= grow // 2
                hi += grow - grow // 2
                if lo < 0:
                    hi -= lo; lo = 0
                if hi > n:
                    lo -= (hi - n); hi = n
                lo = max(0, lo)
            slices.append(slice(int(lo), int(hi)))
        if not per_sample:
            return tuple(slices)
        return self._per_sample_crop(slices, fluence_flat, sizes, unet_multiple)

    def _per_sample_crop(self, shared, fluence_flat, sizes, unet_multiple):
        """Re-cut the shared box's H and W as one window per sample.

        The COMMON SIZE comes from the shared box, which already carries the
        margin, the floor and the multiple-of-16 rounding, and is by
        construction at least as wide as any single aperture in the batch. So
        every sample gets a window of exactly that size, centred on its own
        field -- no sample is cropped tighter than the union path would have
        cropped it, and the boxes stack.
        """
        sd, sh, sw = shared
        size_h, size_w = sh.stop - sh.start, sw.stop - sw.start
        _, Hn, Wn = sizes
        if fluence_flat is None:
            raise ValueError(
                "per-sample cropping needs the projected aperture; centring "
                "on the dose support instead would move the box with the "
                "patient's exit dose rather than with the field.")
        f = fluence_flat.detach()
        peak = f.amax(dim=(-3, -2, -1), keepdim=True)
        sup = (f / peak.clamp_min(1e-12)) > 1e-3
        n = f.shape[0]

        def centres(axis_dim, n_axis, fallback):
            """Per-sample aperture midpoint along one lateral axis.

            Bounds are reduced ON DEVICE and read back in a single transfer.
            The previous form called torch.nonzero plus two float() casts inside
            a per-sample loop -- three device syncs per sample per axis, each
            draining the queue.
            """
            prof = sup.any(dim=1)                       # [N, D, H, W]
            for a in sorted([d for d in (0, 1, 2) if d != axis_dim],
                            reverse=True):
                prof = prof.any(dim=a + 1)              # -> [N, n_axis]
            ar = torch.arange(prof.shape[1], device=prof.device)
            # Sentinels chosen so an all-False row is detectable: lo lands on
            # n_axis, which no real index can take.
            lo = torch.where(prof, ar, torch.full_like(ar, prof.shape[1])).amin(dim=1)
            hi = torch.where(prof, ar, torch.full_like(ar, -1)).amax(dim=1)
            lo_l, hi_l = lo.tolist(), hi.tolist()       # ONE sync for the batch
            return [fallback if lo_l[i] > hi_l[i]       # closed MLC
                    else 0.5 * (lo_l[i] + hi_l[i] + 1)
                    for i in range(n)]

        h_c = centres(1, Hn, 0.5 * (sh.start + sh.stop))
        w_c = centres(2, Wn, 0.5 * (sw.start + sw.stop))

        def starts(cs, size, n_axis):
            out = []
            for c in cs:
                lo = int(round(c - size / 2.0))
                out.append(max(0, min(lo, n_axis - size)))
            return out

        return PerSampleBevCrop(sd, starts(h_c, size_h, Hn),
                                starts(w_c, size_w, Wn), size_h, size_w)

    def _early_bev_work_crop(self, fluence_flat, unet_multiple=16):
        """Crop available before dose convolution, with an exact physics halo.

        ``compute_bev_crop`` first supplies the model box: projected-fluence
        support plus ``bev_crop_margin_mm``.  The water PB and optional
        Gaussian redistribution have finite support.  Only the part of that
        support not already covered by the model margin needs to be added to
        form the physics work box.
        """
        inner = self.compute_bev_crop(
            None, fluence_flat, unet_multiple=unet_multiple)
        sd, sh, sw = inner
        res_h, _res_d, res_w = (float(x) for x in self.dose_grid_spacing)
        pb_h_mm = (self.kernel_size // 2) * res_h
        pb_w_mm = (self.kernel_size // 2) * res_w
        scatter_mm = 0.0
        if self.lateral_scatter:
            if isinstance(self.lat_sigma_mm, (int, float)):
                sigmas = [float(self.lat_sigma_mm) * (1.0 / rho - 1.0)
                          for _lo, _hi, rho in _LAT_BINS]
            else:
                sigmas = [float(x) for x in self.lat_sigma_mm]
            if self.lat_cap_mm is not None:
                sigmas = [min(x, float(self.lat_cap_mm)) for x in sigmas]
            # _gaussian_blur_lateral uses round(3*sigma_vox) on both axes and
            # receives res_h as its lateral resolution.
            scatter_mm = round(3.0 * max(sigmas) / res_h) * res_h
        extra_h = max(0.0, pb_h_mm + scatter_mm - self.bev_crop_margin_mm)
        extra_w = max(0.0, pb_w_mm + scatter_mm - self.bev_crop_margin_mm)
        mh = int(math.ceil(extra_h / max(res_h, 1e-6)))
        mw = int(math.ceil(extra_w / max(res_w, 1e-6)))
        _n, _c, Dz, Hz, Wz = fluence_flat.shape
        work = (
            sd,
            slice(max(0, sh.start - mh), min(Hz, sh.stop + mh)),
            slice(max(0, sw.start - mw), min(Wz, sw.stop + mw)),
        )
        return inner, work

    def _corrector_autocast(self):
        """Autocast context for the corrector forward; a no-op when amp is off."""
        import contextlib
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self.device.type
                                  if hasattr(self.device, "type") else "cuda",
                                  dtype=self.amp_dtype, enabled=True)

    def build_v2_features(self, dose_flat, dens_flat, hu_image,
                          B_, G_, Dz, Hz, Wz, open_area_mm2=None, crop=None):
        """Assemble the v2 BEV feature stack.

        The feature set and the trunk are separate questions: this is kept
        independent of the model so the v1 ``DoseCorrectionModel`` can be fed
        the richer channels (``--v2_features``), which is what the best run
        does -- raw dose in channel 0, then the 12 v2 channels.

        ``crop`` is a ``(slice_d, slice_h, slice_w)`` triple, and the ONE
        invariant it must satisfy is that every emitted value at a voxel inside
        the crop equals its value in the uncropped build. Most channels satisfy
        that automatically -- the position channels are functions of the mm
        coordinate vectors, which are simply sliced. ``cum_electron_path`` does
        NOT: it is a cumulative sum ALONG D, so computing it on a cropped
        volume restarts the integral at the crop face and reports a ray that
        has traversed nothing. It is therefore computed on the FULL volume here
        and passed down pre-sliced via ``cum_override``.
        """
        from features_v2 import build_bev_features, cumulative_path_length
        from materials_v2 import quantize_density, relative_electron_density

        # HU in BEV. Needed because material family is keyed on HU and the
        # HU->density LUT is NOT invertible (it steps down at HU -10->-9 and
        # 120->121), so density cannot recover it. Falls back to the density
        # channel when the caller supplies no HU, which degrades the material
        # embedding to a constant rather than failing.
        if hu_image is not None:
            hu_in = hu_image.unsqueeze(1).permute(0, 1, 3, 2, 4)
            if hu_in.shape[1] == 1 and G_ > 1:
                hu_in = hu_in.expand(-1, G_, -1, -1, -1)
            hu_bev = self.inv_rotation_layer(hu_in).permute(0, 1, 3, 2, 4)
            hu_flat = hu_bev.reshape(B_ * G_, 1, Dz, Hz, Wz).to(dose_flat.dtype)
        else:
            hu_flat = torch.zeros_like(dose_flat)

        flu = getattr(self, "_bev_fluence", None)
        flu_flat = (flu.reshape(B_ * G_, 1, Dz, Hz, Wz).to(dose_flat.dtype)
                    if flu is not None else None)

        depth_mm, lat_h_mm, lat_w_mm = self.bev_axis_coords()
        cum_override = None
        if crop is not None:
            per_sample = isinstance(crop, PerSampleBevCrop)
            cut = crop.apply if per_sample else (
                lambda t: t[..., crop[0], crop[1], crop[2]])
            cfg = self.v2_feature_cfg
            if cfg is None or cfg.use_cumulative_path:
                # Full-volume integral, then sliced. See the docstring.
                dens_for_rho = (quantize_density(dens_flat)
                                if (cfg is None or cfg.quantize_density_to_sim)
                                else dens_flat)
                rho_e_full = relative_electron_density(dens_for_rho, hu_flat)
                cum_full = cumulative_path_length(
                    rho_e_full, float(self.dose_grid_spacing[1]))
                cum_override = cut(cum_full)
            dose_flat = cut(dose_flat)
            dens_flat = cut(dens_flat)
            hu_flat = cut(hu_flat)
            if flu_flat is not None:
                flu_flat = cut(flu_flat)
            if per_sample:
                # [N, size] axes: each sample's own window, so the offset
                # build_bev_features subtracts is its own field centre.
                depth_mm = depth_mm[crop.d]
                lat_h_mm, lat_w_mm = crop.axes(lat_h_mm, lat_w_mm)
            else:
                sd, sh, sw = crop
                depth_mm, lat_h_mm, lat_w_mm = (depth_mm[sd], lat_h_mm[sh],
                                                lat_w_mm[sw])

        built = build_bev_features(
            dose_flat, dens_flat, hu_flat, depth_mm, lat_h_mm, lat_w_mm,
            fluence_bev=flu_flat, open_area_mm2=open_area_mm2,
            sad_mm=float(self.SID), spacing_depth_mm=float(self.dose_grid_spacing[1]),
            cum_override=cum_override,
            cfg=self.v2_feature_cfg,
        )
        return built

    def source_distance_bev(self) -> torch.Tensor:
        """Per-voxel distance to the source in BEV, shape ``[1, G, D, H, W]``.

        Geometric prior for the corrector: at the isocenter the value is
        exactly SAD (~1000 mm); it grows along the beam direction
        (depth) and laterally. Computed in voxel-index coordinates of
        the engine's dose grid, using the assumption that:
          * the BEV depth axis is the engine's D axis (axis 2 in the
            [B,G,D,H,W] dose tensor),
          * the isocenter sits at iso_center / dose_grid_spacing in
            voxel coords,
          * the source is along that depth axis at -SAD from iso.

        Returned distance is normalised by SAD so it's ~1 at iso and
        the lateral falloff is ~O(1) — keeps it in a sane range for the
        first conv before BatchNorm settles.
        """
        H_, D_, W_ = self.dose_grid_shape
        res_h, res_d, res_w = self.dose_grid_spacing
        iso_h_vox = self.iso_center[0] / res_h
        iso_d_vox = self.iso_center[1] / res_d
        iso_w_vox = self.iso_center[2] / res_w
        SAD = float(self.SID)
        h_idx = torch.arange(H_, device=self.device, dtype=torch.float32)
        d_idx = torch.arange(D_, device=self.device, dtype=torch.float32)
        w_idx = torch.arange(W_, device=self.device, dtype=torch.float32)
        # depth-from-source: at iso this is SAD; varies linearly along D.
        depth_mm = (d_idx - iso_d_vox) * res_d + SAD          # along D axis
        lat_h_mm = (h_idx - iso_h_vox) * res_h                # along H axis
        lat_w_mm = (w_idx - iso_w_vox) * res_w                # along W axis
        # Broadcast to [D, H, W] matching the spatial layout of the
        # dose tensor [B, G, D, H, W] used in the corrector input stack.
        dist = torch.sqrt(
            depth_mm.view(-1, 1, 1) ** 2
            + lat_h_mm.view(1, -1, 1) ** 2
            + lat_w_mm.view(1, 1, -1) ** 2
        )  # [D, H, W]
        G = self.number_of_beams
        return (dist / SAD).unsqueeze(0).unsqueeze(0).expand(1, G, -1, -1, -1)

    def patient_to_bev(self, patient_volume: torch.Tensor) -> torch.Tensor:
        """Rotate a patient-frame volume into beam's-eye-view for each beam.

        Accepts:
          * ``[D, H, W]`` — single volume shared across all G beams; the
            inv_rotation_layer broadcasts it.
          * ``[G, D, H, W]`` — one volume per beam (e.g. per-CP target
            dose in batched-CP validation), each rotated through its
            own gantry. G must match the engine's number_of_beams.
          * ``[B, G, D, H, W]`` — pre-shaped batch.

        Returns ``[B, G, D, H, W]`` in BEV coordinates.
        """
        G = self.number_of_beams
        if patient_volume.dim() == 3:
            v = patient_volume.unsqueeze(0).unsqueeze(0)               # [1, 1, D, H, W]
        elif patient_volume.dim() == 4:
            if patient_volume.shape[0] != G:
                # Treat as [B, D, H, W] single-beam-per-sample volume
                # (legacy single-CP path) and add the G=1 axis.
                v = patient_volume.unsqueeze(1)                        # [B, 1, D, H, W]
            else:
                v = patient_volume.unsqueeze(0)                        # [1, G, D, H, W]
        elif patient_volume.dim() == 5:
            v = patient_volume
        else:
            raise ValueError(f"patient_to_bev: unsupported input dim {patient_volume.dim()}")
        # The inv_rotation_layer expects G beams on axis 1 (it was
        # initialised with G gantry angles). When the caller passed a
        # single shared volume (e.g. density), broadcast it to G.
        if v.shape[1] == 1 and G > 1:
            v = v.expand(-1, G, -1, -1, -1)
        x = v.permute(0, 1, 3, 2, 4)
        rotated = self.inv_rotation_layer(x).permute(0, 1, 3, 2, 4)
        return rotated

    def bev_pred_dose(self) -> torch.Tensor | None:
        """Final corrected dose in BEV coordinates, ``[B, G, D, H, W]``.

        Returns ``None`` if forward hasn't been called yet.
        """
        if not hasattr(self, "original_dose") or self.original_dose is None:
            return None
        if getattr(self, "dose_correction", None) is not None:
            return self.original_dose + self.dose_correction
        return self.original_dose

    @property
    def iso_center_voxel(self) -> tuple[int, int, int]:
        if self.iso_center is None:
            return None

        sx, sy, sz = self.dose_grid_shape
        rx, ry, rz = self.dose_grid_spacing
        X, Y, Z = self.iso_center  # physical coords, origin at isocenter corner
        X_center, Y_center, Z_center = (X, Y, Z)

        # Convert physical coords to voxel indices and round to nearest voxel
        ix = int(X_center / rx)
        iy = int(Y_center / ry)
        iz = int(Z_center / rz)

        # Optionally clamp to valid voxel range
        ix = max(0, min(sx - 1, ix))
        iy = max(0, min(sy - 1, iy))
        iz = max(0, min(sz - 1, iz))

        return (ix, iy, iz)


    def _assert_sizes(self, density_image, leaf_positions, jaw_positions, mus, fluence_maps=None):
        """Validate input tensor sizes."""

        G = self.number_of_beams

        if fluence_maps is not None:
            # Derive B from fluence_maps; mus is optional in this path
            fm_h, fm_w = self.field_size
            if fluence_maps.dim() == 4:
                B = fluence_maps.shape[0]
                expected_fm = (B, G, fm_h, fm_w)
                assert fluence_maps.shape == expected_fm, \
                    f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}"
            elif fluence_maps.dim() == 3:
                assert fluence_maps.shape[0] % G == 0, \
                    f"Fluence maps leading dim {fluence_maps.shape[0]} is not divisible by G={G}"
                B = fluence_maps.shape[0] // G
                expected_fm = (B * G, fm_h, fm_w)
                assert fluence_maps.shape == expected_fm, \
                    f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}"
            else:
                raise ValueError(
                    f"fluence_maps must be 3D [B*G, H, W] or 4D [B, G, H, W], got {fluence_maps.dim()}D"
                )

            # Validate mus only when provided
            if mus is not None:
                assert mus.dim() == 2, \
                    f"MUs needs 2 dimensions [B, G], got {mus.dim()}D: {mus.shape}"
                expected_mus = (B, G)
                assert mus.shape == expected_mus, \
                    f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"

            devices = {fluence_maps.device}
            dtypes = {fluence_maps.dtype}
            if mus is not None:
                devices.add(mus.device)
                dtypes.add(mus.dtype)
        else:
            B = leaf_positions.shape[0]
            assert leaf_positions.dim() == 4, \
                f"Leaf positions needs 4 dimensions [B, 2, CP, N], got {leaf_positions.dim()}D: {leaf_positions.shape}"
            assert mus.dim() == 2, \
                f"MUs needs 2 dimensions [B, CP], got {mus.dim()}D: {mus.shape}"

            assert leaf_positions.shape[0] == B and mus.shape[0] == B, \
                f"Batch size mismatch: ct={B}, leaf_positions={leaf_positions.shape[0]}, mus={mus.shape[0]}"

            expected_leaf = (B, G, self.machine_config.number_of_leaf_pairs, 2)
            assert leaf_positions.shape == expected_leaf, \
                f"Leaf positions shape mismatch: expected {expected_leaf}, got {leaf_positions.shape}"

            expected_mus = (B, G)
            assert mus.shape == expected_mus, \
                f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"

            if jaw_positions is not None:
                assert jaw_positions.dim() == 3, \
                    f"Jaw positions needs 3 dimensions [B, 2, CP], got {jaw_positions.dim()}D: {jaw_positions.shape}"

                assert jaw_positions.shape[0] == B, \
                    f"Batch size mismatch: ct={B}, jaw_positions={jaw_positions.shape[0]}"

                expected_jaw = (B, G, 2)
                assert jaw_positions.shape == expected_jaw, \
                    f"Jaw positions shape mismatch: expected {expected_jaw}, got {jaw_positions.shape}"

            devices = {leaf_positions.device, mus.device}
            if jaw_positions is not None:
                devices.add(jaw_positions.device)
            dtypes = {leaf_positions.dtype, mus.dtype}
            if jaw_positions is not None:
                dtypes.add(jaw_positions.dtype)

        if density_image is None:
            raise ValueError("CT image must be provided.")
        assert density_image.dim() == 4, \
            f"CT image needs 4 dimensions [B, D, H, W], got {density_image.dim()}D: {density_image.shape}"

        expected_ct = (B, *self.dose_grid_shape)
        assert density_image.shape == expected_ct, \
            f"CT shape mismatch: expected {expected_ct}, got {density_image.shape}"

        devices.add(density_image.device)
        dtypes.add(density_image.dtype)

        if len(devices) != 1:
            raise ValueError(f"Device mismatch among tensors: {devices}")

        if len(dtypes) != 1:
            raise ValueError(f"Dtype mismatch among tensors: {dtypes}")
        
        
    def forward(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        support_image: torch.Tensor | None = None,
        return_per_beam: bool = False,
        absorption_scale: torch.Tensor | None = None,
        hu_image: torch.Tensor | None = None,
        open_area_mm2: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """
        Runs the full dose calculation pipeline.

        Args:
            leaf_positions: Leaf positions [B, G, N, 2]. Not required when fluence_maps is provided.
            mus: Monitor units [B, G]. Optional when fluence_maps is provided; if supplied the dose
                is scaled by MUs, if omitted the fluence maps are used as-is.
            jaw_positions: Jaw positions [B, G, 2]. Not required when fluence_maps is provided.
            density_image: CT image tensor [B, D, H, W].
            support_image: Support image tensor [B, D, H, W].
            fluence_maps: Optional pre-computed fluence maps [B, G, H, W] or [B*G, H, W].
                If provided, the FluenceMapLayer is skipped and leaf_positions/jaw_positions
                are ignored. The maps are used directly as input to the FluenceVolumeLayer.
            absorption_scale: Optional per-voxel multiplier [B, D, H, W] in PATIENT space,
                applied to the final dose. Its purpose is the dose-to-water -> dose-to-medium
                conversion: every transport path in here deposits a WATER kernel at
                radiological depth, i.e. it produces dose-to-water, whereas Geant4 scores
                dose-to-medium. The ratio is (mu_en/rho)_mat / (mu_en/rho)_water, which at
                this spectrum is (Z/A)_mat / (Z/A)_water -- see
                ``materials_v2.relative_zoa_from_hu``. None (default) leaves v1 behaviour
                untouched. Applied AFTER the corrector, so the corrector still sees
                dose-to-water.

        Returns:
            Dose tensor [B, D, H, W].
        """
        self._set_device_dtype(leaf_positions.device, leaf_positions.dtype)

        if not self.layers_initialized:
            raise Exception("Layers haven't been initialized yet. Dose engine cannot perform dose calculations.")

        self._assert_sizes(density_image, leaf_positions, jaw_positions, mus, fluence_maps=None)

        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if not self.fluence_to_dose:
                with torch.no_grad():
                    batched_radiological_depths = self.rad_depth_layer(density_image).detach()
                    batched_kernels = self.pencil_beam_kernel_layer(batched_radiological_depths).detach()
                del batched_radiological_depths
            H, D, W = self.dose_grid_shape

            G = self.number_of_beams

            batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)
            self.original_fluence_maps = batched_fluence_maps.unsqueeze(1).detach()
            B = leaf_positions.shape[0]


            if self.fluence_correction_model is not None:
                if self.fluence_correction_grad:
                    self.fluence_map_correction = self.fluence_correction_model(batched_fluence_maps.unsqueeze(1))[:, 0, :, :]
                else:
                    with torch.no_grad():
                        self.fluence_map_correction = self.fluence_correction_model(
                            batched_fluence_maps.unsqueeze(1))[:, 0, :, :]
                batched_fluence_maps = self.fluence_map_correction + batched_fluence_maps
            else:
                self.fluence_map_correction = None

            # Apply collimator rotation (beam limiting device angle)
            # This rotates the fluence map in-plane before projection to 3D
            if self._has_collimator_rotation:
                batched_fluence_maps = rotate_2d_images(
                    batched_fluence_maps,
                    self.collimator_angles,
                    device=self.device,
                    dtype=self.dtype
                )  # [B*G, H, W]

            batched_fluence_volumes = self.fluence_volume_layer(
                batched_fluence_maps
            )
            # Optional TERMA correction: scale the divergent source volume
            # BEFORE water-kernel convolution. The existing multislab path
            # correction remains downstream and uses the original density.
            bev_density = None
            if self.terma_scaling:
                with torch.no_grad():
                    bev_density = self.patient_to_bev(density_image)
                terma_scale = self.terma_scaling_layer(
                    batched_fluence_maps, bev_density)
                batched_fluence_volumes = batched_fluence_volumes * terma_scale.to(
                    device=batched_fluence_volumes.device,
                    dtype=batched_fluence_volumes.dtype,
                )
                self.terma_scale = terma_scale.detach()
                self.terma_field_size_mm = (
                    self.terma_scaling_layer.last_field_size_mm.detach())
            else:
                self.terma_scale = None
                self.terma_field_size_mm = None

            # The projected fluence is the earliest rotation-safe field
            # support. Keep the full copy for feature construction and derive
            # an optional finite-support work crop before the expensive PB
            # convolution. The projector itself is still full-grid in this
            # first implementation; its bbox hook can use these slices once
            # the bounds are computed directly from the 2-D map.
            physics_crop = None
            lattice_applied_residual = False
            self.early_bev_crop_slices = None
            if True:
                _fv_full = batched_fluence_volumes.squeeze(-1)
                self._bev_fluence = _fv_full.view(
                    B, G, _fv_full.shape[-3], _fv_full.shape[-2],
                    _fv_full.shape[-1]).detach()
                if self.early_bev_crop:
                    _mult = 2 ** getattr(
                        getattr(self.dose_correction_model, "trunk", None),
                        "depth", 4)
                    _inner, physics_crop = self._early_bev_work_crop(
                        _fv_full.reshape(
                            B * G, 1, _fv_full.shape[-3],
                            _fv_full.shape[-2], _fv_full.shape[-1]),
                        unet_multiple=_mult)
                    self.early_bev_crop_slices = physics_crop
                    _sd, _sh, _sw = physics_crop
                    batched_fluence_volumes = batched_fluence_volumes[
                        :, _sd, _sh, _sw, :]
                    if not self.fluence_to_dose:
                        batched_kernels = batched_kernels[..., _sd]
                del _fv_full
            if self.fluence_to_dose:
                # No kernel convolution: the fluence volume itself is the CNN input.
                model_input_vol = batched_fluence_volumes
                del batched_fluence_volumes, batched_fluence_maps
            elif self.lattice_size >= 1:
                # Multilattice: one ray per equal-fluence tile instead of
                # one through the isocentre. The per-tile multislab
                # residual is applied INSIDE this call, against each tile's
                # own ray, so `lattice_applied_residual` below suppresses
                # the single-ray `cf` that would otherwise double-correct.
                if bev_density is None:
                    bev_density = self.patient_to_bev(density_image)
                _rd_full = divergent_radiological_depth(
                    bev_density, self.SID, self.dose_grid_spacing,
                    self.iso_center)
                _rd = _rd_full
                _density = bev_density
                if physics_crop is not None:
                    _sd, _sh, _sw = physics_crop
                    _rd = _rd_full[..., _sd, _sh, _sw]
                    _density = bev_density[..., _sd, _sh, _sw]
                model_input_vol = self.lattice_pencil_beam(
                    batched_fluence_volumes,
                    _rd.reshape(B * G, 1, *_rd.shape[-3:]),
                    _density.reshape(B * G, 1, *_density.shape[-3:]),
                    physics_crop, tile_chunk=self.lattice_tile_chunk)
                lattice_applied_residual = True
                del batched_fluence_volumes, batched_fluence_maps
                del batched_kernels, _rd, _rd_full, _density
            else:
                model_input_vol = self.beam_wise_conv_layer(
                    batched_fluence_volumes, batched_kernels
                )
                del batched_fluence_volumes, batched_fluence_maps, batched_kernels
            model_input_vol = model_input_vol * self.machine_config.mean_photon_energy_MeV

            D_, H_, W_, _ = model_input_vol.shape[1:]
            model_input_vol = model_input_vol.view(B, G, D_, H_, W_)
            if mus is not None:
                model_input_vol = model_input_vol * mus[:, :, None, None, None]

            batched_accumulated_dose = model_input_vol
            self.original_dose = model_input_vol.detach()

            def _scatter_early_crop(volume):
                if physics_crop is None:
                    return volume
                _sd, _sh, _sw = physics_crop
                full = volume.new_zeros((B, G, D, H, W))
                full[..., _sd, _sh, _sw] = volume
                return full

            # ---- Multislab heterogeneity physics --------------------------------
            # Hoisted OUT of the corrector branch: this is pure physics on the
            # pencil-beam baseline (per-voxel radiological-depth attenuation, plus
            # optional density-scaled lateral scatter), and it has to be available
            # with dose_correction_model=None so the engine can be commissioned
            # against the reference on its own.
            rad_depth = None
            rotated_density_image = None
            if self.multislab and not self.fluence_to_dose:
                if bev_density is None:
                    bev_density = self.patient_to_bev(density_image)
                rad_depth = divergent_radiological_depth(
                    bev_density, self.SID, self.dose_grid_spacing, self.iso_center)
                ih = int(min(max(
                    self.iso_center[0] / self.dose_grid_spacing[0], 0), H - 1))
                iw = int(min(max(
                    self.iso_center[2] / self.dose_grid_spacing[2], 0), W - 1))
                central_rad = rad_depth[:, :, :, ih:ih + 1, iw:iw + 1]
                if physics_crop is None:
                    physics_rad = rad_depth
                    physics_density = bev_density
                else:
                    _sd, _sh, _sw = physics_crop
                    physics_rad = rad_depth[..., _sd, _sh, _sw]
                    physics_density = bev_density[..., _sd, _sh, _sw]
                    central_rad = central_rad[..., _sd, :, :]
                if not lattice_applied_residual:
                    cf = torch.exp(
                        -self.mu_eff * (physics_rad - central_rad)).clamp(0.3, 3.0)
                    batched_accumulated_dose = batched_accumulated_dose * cf
                # else: the lattice already applied a residual PER TILE, each
                # against its own ray. Multiplying by the single-ray cf here
                # would correct the same heterogeneity twice, and the result
                # would still look like a plausible dose.
                # Cached so a commissioning fit can vary the lateral-scatter
                # parameters without recomputing the whole pencil-beam forward.
                self.bev_density = bev_density.detach()
                self.bev_dose_prescatter = _scatter_early_crop(
                    batched_accumulated_dose).detach()
                if self.lateral_scatter:
                    fn = (lateral_scatter_correction_fine if self.lat_fine
                          else lateral_scatter_correction)
                    batched_accumulated_dose = fn(
                        batched_accumulated_dose, physics_density,
                        self.lat_sigma_mm, self.dose_grid_spacing[0], self.lat_cap_mm)
                if self.lateral_scatter_model is not None:
                    scatter_fluence = self._bev_fluence
                    if physics_crop is not None:
                        _sd, _sh, _sw = physics_crop
                        scatter_fluence = scatter_fluence[..., _sd, _sh, _sw]
                    if self.lateral_scatter_grad:
                        batched_accumulated_dose = self.lateral_scatter_model(
                            batched_accumulated_dose, physics_density,
                            scatter_fluence, self.dose_grid_spacing[0])
                    else:
                        with torch.no_grad():
                            batched_accumulated_dose = self.lateral_scatter_model(
                                batched_accumulated_dose, physics_density,
                                scatter_fluence, self.dose_grid_spacing[0])
                batched_accumulated_dose = _scatter_early_crop(
                    batched_accumulated_dose)
                self.original_dose = batched_accumulated_dose.detach()
                if support_image is None:
                    rotated_density_image = bev_density
            elif physics_crop is not None:
                batched_accumulated_dose = _scatter_early_crop(
                    batched_accumulated_dose)
                self.original_dose = batched_accumulated_dose.detach()

            if self.dose_correction_model is not None:
                if support_image is None:
                    support_image = density_image
                # Rotate the density image into BEV for each beam,
                # producing shape [B, G, D, H, W]. The rotation layer
                # was built with G gantry angles, so the density (one
                # volume shared across beams) must be broadcast to G
                # BEFORE the rotation call.
                support_in = support_image.unsqueeze(1).permute(0, 1, 3, 2, 4)  # [B, 1, H, D, W]
                if support_in.shape[1] == 1 and G > 1:
                    support_in = support_in.expand(-1, G, -1, -1, -1)
                if rotated_density_image is None or support_image is not density_image:
                    rotated_density_image = self.inv_rotation_layer(support_in).permute(0, 1, 3, 2, 4)

                # Fold the beam dimension into the batch so the 3D model is
                # always called with a fixed number of input channels, independent
                # of how many beams the engine handles.
                B_, G_, Dz, Hz, Wz = batched_accumulated_dose.shape

                # rad_depth (if multislab is on) was computed above as pure physics;
                # here it just becomes an extra corrector input channel.
                dose_flat = batched_accumulated_dose.reshape(B_ * G_, 1, Dz, Hz, Wz)
                dens_flat = rotated_density_image.reshape(B_ * G_, 1, Dz, Hz, Wz)
                feats = [dose_flat, dens_flat]
                if rad_depth is not None:                              # radiological-depth channel
                    feats.append(rad_depth.reshape(B_ * G_, 1, Dz, Hz, Wz).to(dose_flat.dtype))
                if self.use_source_distance:                          # geometric PSF prior (r / SAD)
                    psf_bev = self.source_distance_bev()               # [1, G, D, H, W]
                    psf_flat = psf_bev.expand(B_, -1, -1, -1, -1).reshape(B_ * G_, 1, Dz, Hz, Wz)
                    feats.append(psf_flat.to(dose_flat.dtype))
                # ---- BEV crop -------------------------------------------
                # Between the rotation and the corrector, which is the only
                # place it is both correct and worth doing (see __init__).
                crop = None
                self.bev_crop_slices = None
                if self.bev_crop:
                    if not self.v2_features:
                        raise ValueError(
                            "bev_crop currently requires v2_features: the "
                            "legacy channel path builds its inputs from "
                            "full-volume tensors that are not sliced here.")
                    _mult = 2 ** getattr(
                        getattr(self.dose_correction_model, "trunk", None),
                        "depth", 4)
                    crop = self.compute_bev_crop(
                        dose_flat,
                        (self._bev_fluence.reshape(B_ * G_, 1, Dz, Hz, Wz)
                         if getattr(self, "_bev_fluence", None) is not None else None),
                        unet_multiple=_mult,
                        per_sample=self.bev_crop_per_sample)
                    self.bev_crop_slices = crop
                    self._dose_for_crop = dose_flat.detach()
                    # Report the ACTUAL saving once per process. Every figure
                    # for this before it ran was an estimate built on one
                    # measured box (86x278x295) plus assumed field sizes, and
                    # the batch union makes it very sensitive to how far the
                    # arc's apertures wander -- SameBeamBatchSampler shuffles
                    # WITHIN a beam, so the 4 CPs of a batch are random gantry
                    # angles, not neighbours.
                    if not getattr(CorrectedDoseEngine, "_crop_reported", False):
                        CorrectedDoseEngine._crop_reported = True
                        _cd, _ch, _cw = (crop.shape
                                         if isinstance(crop, PerSampleBevCrop)
                                         else tuple(s.stop - s.start for s in crop))
                        _before, _after = Dz * Hz * Wz, _cd * _ch * _cw
                        print(f"[bev_crop] {Dz}x{Hz}x{Wz} -> {_cd}x{_ch}x{_cw}  "
                              f"({_before/1e6:.2f}M -> {_after/1e6:.2f}M voxels, "
                              f"{_before/max(_after,1):.2f}x, margin "
                              f"{self.bev_crop_margin_mm:.0f} mm)", flush=True)

                material_flat = None
                if self.v2_features:
                    # Raw dose stays channel 0: v1 reads x[:, 0:1] for its gain
                    # term and the engine adds the result back in physical
                    # units, so the peak-normalised channel 0 that
                    # build_v2_features emits cannot take that slot.
                    _built = self.build_v2_features(
                        dose_flat, dens_flat, hu_image, B_, G_, Dz, Hz, Wz,
                        open_area_mm2=open_area_mm2, crop=crop)
                    # Channel 0 is the RAW dose in physical units, so it has to
                    # be sliced to match the (already cropped) feature stack.
                    if crop is None:
                        _dose_ch = dose_flat
                    elif isinstance(crop, PerSampleBevCrop):
                        _dose_ch = crop.apply(dose_flat)
                    else:
                        _dose_ch = dose_flat[..., crop[0], crop[1], crop[2]]
                    stack_images = torch.cat(
                        [_dose_ch, _built["features"].to(dose_flat.dtype)], dim=1)
                    material_flat = _built["material_id"]
                else:
                    stack_images = torch.cat(feats, dim=1)
                if isinstance(self.dose_correction_model, LateralTTA):
                    res_h, _res_d, res_w = self.dose_grid_spacing
                    self.dose_correction_model.set_centers(
                        self.iso_center[0] / res_h, self.iso_center[2] / res_w)
                # Only the v1 trunk takes material_id, and only when it was
                # built with an embedding. Passing it unconditionally would
                # break LateralTTA and the transformer, both of which are
                # plain tensor-in/tensor-out.
                _needs_material = getattr(
                    self.dose_correction_model, "material_embedding", None) is not None
                # Masked norms need the projected aperture on the SAME grid the
                # stack is on, i.e. already cropped when the crop is on.
                _kw = {}
                if getattr(self.dose_correction_model,
                           "wants_aperture_mask", False):
                    _kw["aperture_mask"] = self._aperture_mask_bev(
                        B_, G_, Dz, Hz, Wz, crop, stack_images.dtype)
                with self._corrector_autocast():
                    if _needs_material:
                        out_flat = self.dose_correction_model(
                            stack_images, material_id=material_flat, **_kw)
                    else:
                        out_flat = self.dose_correction_model(stack_images, **_kw)
                # Deep supervision: aux predictions are CORRECTED DOSE in BEV at
                # reduced resolution. They are stashed rather than returned
                # because forward()'s contract is one dose tensor; the training
                # loop reads engine.dose_correction_deep.
                self.dose_correction_deep = ()
                if isinstance(out_flat, dict):
                    self.dose_correction_deep = tuple(out_flat.get("deep_supervision", ()))
                    out_flat = out_flat["correction"]
                out_flat = out_flat.to(stack_images.dtype)
                if crop is not None:
                    # Scatter back to the full BEV grid. The correction is
                    # exactly ZERO outside the crop, which is a real modelling
                    # assumption and not just bookkeeping: the corrector can no
                    # longer touch anything beyond the margin. It is a small
                    # assumption only because the bounded+relu'd head already
                    # emits almost nothing out there (leak_mass 4e-06 of the
                    # true dose), and the margin is sized to cover the scatter
                    # halo -- but it IS what --bev_crop_margin_mm buys.
                    if isinstance(crop, PerSampleBevCrop):
                        out_flat = crop.scatter(out_flat, (B_ * G_, 1, Dz, Hz, Wz))
                    else:
                        full = out_flat.new_zeros((B_ * G_, 1, Dz, Hz, Wz))
                        full[..., crop[0], crop[1], crop[2]] = out_flat
                        out_flat = full
                out = out_flat.reshape(B_, G_, Dz, Hz, Wz)
                if self.fluence_to_dose:
                    # the CNN output IS the dose (channel 0 was the fluence volume);
                    # there is no physics baseline to add to.
                    self.dose_correction = None
                    batched_accumulated_dose = out
                    self.original_dose = out.detach()
                else:
                    self.dose_correction = out
                    batched_accumulated_dose = out + batched_accumulated_dose
            else:
                self.dose_correction = None
                self.dose_correction_deep = ()

            batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)

            if absorption_scale is not None:
                # [B, D, H, W] -> broadcast over the beam dimension of [B, G, D, H, W].
                batched_accumulated_dose = batched_accumulated_dose * absorption_scale.unsqueeze(1).to(
                    batched_accumulated_dose.dtype)

            # Keep the G beam dimension when the caller wants per-CP
            # supervision (real-batching training). Default behaviour
            # (return_per_beam=False) sums G into one plan-level dose,
            # which is what every existing single-CP caller expects.
            if return_per_beam:
                batched_accumulated_dose = batched_accumulated_dose.to(self.dtype)
            else:
                batched_accumulated_dose = batched_accumulated_dose.sum(dim=1).to(self.dtype)

        return batched_accumulated_dose

    def compute_dose(
        self,
        beam_input: BeamSequence | Beam,
        density_image: torch.Tensor | None = None,
        return_intermediates: bool = False,
        overwrite: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute dose from a BeamSequence or Beam.

        Args:
            beam_input: BeamSequence (shapes: mus [CP], leaf_positions [CP, N, 2], jaw_positions [CP, 2])
                or a single Beam. Always required for geometry (gantry angles, iso center, etc.).
            density_image: CT image tensor [1, D, H, W].
            return_intermediates: If True, also return intermediate tensors.
            overwrite: Re-initialize layers even if already set up.
            fluence_maps: Optional pre-computed fluence maps [1, G, H, W] or [G, H, W].
                If provided, the FluenceMapLayer is skipped and leaf/jaw positions from
                beam_input are ignored. G must equal the number of beams in beam_input.

        Returns:
            Dose tensor [1, D, H, W].
        """
        self._initialize_layers(beam_input, overwrite)

        # Add batching dimension to parameters
        if density_image is not None:
            ct_tensor = density_image
            if ct_tensor.dim() == 3:
                ct_tensor = ct_tensor.unsqueeze(0)
        else:
            ct_tensor = None

        if isinstance(beam_input, Beam):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0).unsqueeze(0)
            mus = beam_input.mu.unsqueeze(0).unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0).unsqueeze(0)
        elif isinstance(beam_input, BeamSequence):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0)
            mus = beam_input.mus.unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0)

        # Normalise fluence_maps to [1, G, H, W] so forward() can reshape to [B*G, H, W]
        if fluence_maps is not None and fluence_maps.dim() == 3:
            fluence_maps = fluence_maps.unsqueeze(0)  # [G, H, W] -> [1, G, H, W]

        return self.forward(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            density_image=ct_tensor,
            return_intermediates=return_intermediates,
            fluence_maps=fluence_maps,
        )

    def compute_dose_sequential(
        self,
        beam_sequence: BeamSequence,
        density_image: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute dose by processing beams sequentially or in batches (memory efficient).

        Args:
            beam_sequence: BeamSequence containing all control points
            density_image: CT image tensor [1, D, H, W]

        Returns:
            Accumulated dose tensor [1, H, D, W]
        """
        self._initialize_layers(beam_sequence)
        total_dose = None

        # Process beams one by one
        for beam in beam_sequence:
            beam_dose = self.compute_dose(
                beam,
                density_image=density_image,
                overwrite=True
            )

            if total_dose is None:
                total_dose = beam_dose
            else:
                total_dose = total_dose + beam_dose

        self._initialize_layers(beam_sequence, overwrite=True)
        return total_dose

    def calibrate(self,                   
                  calibration_mu: float = None,
                  original_beam_template: BeamSequence | None = None, # Deprecated setting
                  verbose: bool = True) -> None:
        """
        Calibrates the model by normalizing the output dose so that the pre-defined MU value corresponds to 1Gy.

        Args:
            calibration_mu: The MU where the delivered dose should correspond to 1Gy in water at 10cm depth.
            original_beam_template: A depracated argument for setting the template back to the engine.
            verbose: Enable verbose output (default: False).

        Returns:
            None
        """
        if self.machine_config is None:
            raise Exception("machine_config must be set before calibration.")
        if self.dose_grid_shape is None:
            raise Exception("dose_grid_shape must be set before calibration.")
        if self.dose_grid_spacing is None:
            raise Exception("dose_grid_spacing must be set before calibration.")

        # Apply defaults so calibration works even without a prior beam template
        if self.device is None:
            self.device = torch.device('cpu')
        if self.dtype is None:
            self.dtype = torch.float32
            
        if original_beam_template is not None:
            print("The argument `original_beam_template` is now deprecated and will not be used for calibration")
        center_x, _, center_z = torch.tensor(self.dose_grid_spacing) * (torch.tensor(self.dose_grid_shape)) / 2
        iso_center = (center_x.item(), 100.0, center_z.item())
        beam = Beam.create(0.0, self.machine_config.number_of_leaf_pairs, 0.0, (100.0, 100.0), iso_center=iso_center, device=self.device, dtype=self.dtype)
        if calibration_mu is None:
            calibration_mu = self.machine_config.calibration_mu

        beam.mu = calibration_mu * beam.mu
        water_attenuation = torch.ones(self.dose_grid_shape).to(self.device).to(self.dtype)

        self.layers_initialized = False

        dose = self.compute_dose(
            beam,
            density_image=water_attenuation,
            overwrite=True
        )

        # Get center dose (at 10cm depth - index 50 for 100 voxels)
        center_dose = dose[0, *self.iso_center_voxel].detach().cpu().numpy().item()

        # Calculate calibration factor
        # This gives the factor to normalize to 1 Gy per MU at reference conditions
        calibration_factor = self.machine_config.mean_photon_energy_MeV / center_dose

        if abs(center_dose - 1.0) > 0.001:
            if verbose:
                print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
            self.machine_config.mean_photon_energy_MeV = calibration_factor

        # Reset layers to apply new beam sequence
        self.layers_initialized = False


def load_corrector(weights_path, device, verbose=True):
    """Load a DoseCorrectionModel checkpoint, INFERRING its architecture.

    Sweep variants differ in refine / attention / residual / capacity, so a
    hard-coded architecture either fails a strict load or, worse, silently loads
    a mismatched net. Read what the tensor shapes give directly, then search the
    small remaining flag space for the combination that reproduces the state_dict
    key-for-key and shape-for-shape.
    """
    import itertools
    import os as _os
    ckpt = torch.load(weights_path, map_location=device)
    sd = ckpt["dose_model_state_dict"]
    target = {k: tuple(v.shape) for k, v in sd.items()}
    first = next(v for k, v in sd.items() if k.endswith("weight") and v.dim() == 5)
    out_ch, in_ch = int(first.shape[0]), int(first.shape[1])

    for att, refine, residual, gain, depth in itertools.product(
            ("none", "se", "sa"), (True, False), (True, False), (True, False), (4, 5, 3)):
        try:
            m = DoseCorrectionModel(in_channels=in_ch, base_channels=out_ch, depth=depth,
                                    use_gain=gain, attention=att, refine=refine,
                                    residual=residual)
        except Exception:
            continue
        if {k: tuple(v.shape) for k, v in m.state_dict().items()} == target:
            m.load_state_dict(sd, strict=True)
            if verbose:
                print(f"[model] {_os.path.basename(str(weights_path))}: in_channels={in_ch} "
                      f"base_channels={out_ch} depth={depth} attention={att!r} "
                      f"refine={refine} residual={residual} use_gain={gain} "
                      f"({sum(p.numel() for p in m.parameters()):,} params)", flush=True)
                for k in ("experiment_name", "epoch", "best_val_plan_gamma_1_1",
                          "best_val_lv1_beam_mae"):
                    if k in ckpt:
                        print(f"[model]   {k}: {ckpt[k]}", flush=True)
            return m.to(device).to(torch.float32).eval()

    raise RuntimeError(
        f"Could not infer the architecture of {weights_path}: nothing reproduces its "
        f"state_dict with in_channels={in_ch}, base_channels={out_ch}.")


def warm_start_from(model, weights_path, verbose=True):
    """Load every tensor from a checkpoint that MATCHES this model by name+shape.

    Sweep variants change the architecture (attention, capacity, depth), so a
    strict load is impossible and a silent partial load is dangerous. This copies
    what fits, leaves the rest at its initialisation, and reports both counts so
    the log records exactly how much was inherited.
    """
    ckpt = torch.load(weights_path, map_location="cpu")
    src = ckpt.get("dose_model_state_dict", ckpt)
    tgt = model.state_dict()
    take = {k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}
    missing = [k for k in tgt if k not in take]
    dropped = [k for k in src if k not in take]
    tgt.update(take)
    model.load_state_dict(tgt)
    if verbose:
        n_take = sum(v.numel() for v in take.values())
        n_tot = sum(v.numel() for v in tgt.values())
        print(f"[warm-start] {weights_path}: inherited {len(take)}/{len(tgt)} tensors "
              f"({100.0 * n_take / max(n_tot, 1):.1f}% of parameters)", flush=True)
        if ckpt.get("experiment_name"):
            print(f"[warm-start]   source run: {ckpt['experiment_name']} "
                  f"epoch {ckpt.get('epoch')} "
                  f"gamma {ckpt.get('best_val_plan_gamma_1_1')}", flush=True)
        if missing:
            print(f"[warm-start]   randomly initialised ({len(missing)}): "
                  f"{missing[:4]}{' ...' if len(missing) > 4 else ''}", flush=True)
        if dropped:
            print(f"[warm-start]   unused from checkpoint ({len(dropped)}): "
                  f"{dropped[:4]}{' ...' if len(dropped) > 4 else ''}", flush=True)
    return len(take), len(missing)
