"""Tiny learned mixture over fixed lateral-scatter operators."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from pydosert.engine.multislab_engine import _gaussian_blur_lateral


class TinyLateralScatterMixture(nn.Module):
    """Identity-initialised, anatomy-conditioned lateral redistribution.

    The network does not predict dose.  It chooses a bounded signed fraction of
    the local dose and mixes fixed Gaussian redistribution operators.  Every
    basis term has the conservative form ``blur(source) - source``.  A zeroed
    fraction head therefore reproduces the input dose exactly.
    """

    def __init__(self, hidden_channels: int = 4,
                 sigmas_mm=(2.0, 5.0, 10.0), max_fraction: float = 0.4,
                 depth_spacing_mm: float = 2.0, path_scale_mm: float = 400.0):
        super().__init__()
        h = int(hidden_channels)
        if h < 2:
            raise ValueError("hidden_channels must be >= 2")
        if not sigmas_mm or any(float(s) <= 0.0 for s in sigmas_mm):
            raise ValueError("sigmas_mm must contain positive values")
        if not 0.0 <= max_fraction <= 1.0:
            raise ValueError("max_fraction must lie in [0,1]")
        self.sigmas_mm = tuple(float(s) for s in sigmas_mm)
        self.max_fraction = float(max_fraction)
        self.depth_spacing_mm = float(depth_spacing_mm)
        self.path_scale_mm = float(path_scale_mm)
        self.features = nn.Sequential(
            nn.Conv3d(4, h, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, h, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, h, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.SiLU(inplace=True),
        )
        # The zero bias gives an exact identity at initialisation while also
        # allowing the first optimisation steps to learn a global,
        # density-gated redistribution.  Without it, the signal has to pass
        # through three randomly initialised feature convolutions and the
        # initial gradient is needlessly tiny.
        self.fraction_head = nn.Conv3d(h, 1, 1, bias=True)
        self.mixture_head = nn.Conv3d(h, len(self.sigmas_mm), 1)
        nn.init.zeros_(self.fraction_head.weight)
        nn.init.zeros_(self.fraction_head.bias)
        nn.init.zeros_(self.mixture_head.weight)
        nn.init.zeros_(self.mixture_head.bias)
        self.last_fraction: torch.Tensor | None = None
        self.last_mixture: torch.Tensor | None = None

    def forward(self, dose_bev: torch.Tensor, bev_density: torch.Tensor,
                bev_fluence: torch.Tensor, res_lat_mm: float) -> torch.Tensor:
        if dose_bev.ndim != 5:
            raise ValueError("dose_bev must be [B,G,D,H,W]")
        if bev_density.shape != dose_bev.shape or bev_fluence.shape != dose_bev.shape:
            raise ValueError("density and fluence must match dose_bev")
        b, g, d, h, w = dose_bev.shape
        n = b * g
        rho = bev_density.reshape(n, d, h, w).to(dose_bev.dtype)
        dose = dose_bev.reshape(n, d, h, w)
        fluence = bev_fluence.reshape(n, d, h, w).to(dose_bev.dtype)
        dose_n = dose / dose.detach().amax(
            dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
        fluence_n = fluence / fluence.detach().amax(
            dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
        path = torch.cumsum(rho.clamp_min(0.0), dim=1)
        path = path * self.depth_spacing_mm / max(self.path_scale_mm, 1e-6)
        x = torch.stack((
            dose_n.clamp(0.0, 2.0),
            fluence_n.clamp(0.0, 2.0),
            rho.clamp(0.0, 3.0) / 1.5 - 0.5,
            path.clamp(0.0, 2.0),
        ), dim=1)
        feat = self.features(x)
        # Smoothly suppress external air and near-water/tissue.  This leaves
        # the learned operator focused on lung and other low-density material.
        material_gate = (torch.sigmoid((rho - 0.02) * 40.0)
                         * torch.sigmoid((0.92 - rho) * 12.0))
        fraction = (self.max_fraction * torch.tanh(self.fraction_head(feat)[:, 0])
                    * material_gate)
        mixture = torch.softmax(self.mixture_head(feat), dim=1)
        self.last_fraction = fraction
        self.last_mixture = mixture

        base = dose_bev
        correction = torch.zeros_like(base)
        for i, sigma_mm in enumerate(self.sigmas_mm):
            source = (dose * fraction * mixture[:, i]).reshape(b, g, d, h, w)
            spread = _gaussian_blur_lateral(source, sigma_mm / float(res_lat_mm))
            correction = correction + spread - source
        return (base + correction).clamp_min(0.0)


class TinyLateralHeterogeneityCorrection(nn.Module):
    """Very small correction focused on lateral material interfaces.

    Unlike :class:`TinyLateralScatterMixture`, this model does not spend learned
    convolutions discovering density interfaces.  Signed finite differences of
    density along both lateral axes are supplied explicitly and a pointwise
    network only selects how much of each fixed, conservative scatter basis to
    apply.  There is no learned depth or lateral convolution.
    """

    def __init__(self, hidden_channels: int = 4,
                 sigmas_mm=(2.0, 5.0, 10.0), max_fraction: float = 0.4,
                 depth_spacing_mm: float = 2.0, path_scale_mm: float = 400.0):
        super().__init__()
        h = int(hidden_channels)
        if h < 2:
            raise ValueError("hidden_channels must be >= 2")
        if not sigmas_mm or any(float(s) <= 0.0 for s in sigmas_mm):
            raise ValueError("sigmas_mm must contain positive values")
        if not 0.0 <= max_fraction <= 1.0:
            raise ValueError("max_fraction must lie in [0,1]")
        self.sigmas_mm = tuple(float(s) for s in sigmas_mm)
        self.max_fraction = float(max_fraction)
        self.depth_spacing_mm = float(depth_spacing_mm)
        self.path_scale_mm = float(path_scale_mm)
        # Six explicit local features -> four latent values is only 28
        # parameters at the default width. The heads bring the complete model
        # to 48 parameters for three redistribution radii.
        self.features = nn.Sequential(
            nn.Conv3d(6, h, 1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.fraction_head = nn.Conv3d(h, 1, 1, bias=True)
        self.mixture_head = nn.Conv3d(h, len(self.sigmas_mm), 1, bias=True)
        nn.init.zeros_(self.fraction_head.weight)
        nn.init.zeros_(self.fraction_head.bias)
        nn.init.zeros_(self.mixture_head.weight)
        nn.init.zeros_(self.mixture_head.bias)
        self.last_fraction: torch.Tensor | None = None
        self.last_mixture: torch.Tensor | None = None

    @staticmethod
    def _lateral_density_differences(rho: torch.Tensor):
        # Backward differences retain the input shape and keep the sign, which
        # distinguishes entering from leaving a low-density region.
        dh = F.pad(rho[..., 1:, :] - rho[..., :-1, :], (0, 0, 1, 0))
        dw = F.pad(rho[..., 1:] - rho[..., :-1], (1, 0, 0, 0))
        return dh, dw

    def forward(self, dose_bev: torch.Tensor, bev_density: torch.Tensor,
                bev_fluence: torch.Tensor, res_lat_mm: float) -> torch.Tensor:
        if dose_bev.ndim != 5:
            raise ValueError("dose_bev must be [B,G,D,H,W]")
        if bev_density.shape != dose_bev.shape or bev_fluence.shape != dose_bev.shape:
            raise ValueError("density and fluence must match dose_bev")
        b, g, d, h, w = dose_bev.shape
        n = b * g
        dose = dose_bev.reshape(n, d, h, w)
        rho = bev_density.reshape(n, d, h, w).to(dose_bev.dtype)
        fluence = bev_fluence.reshape(n, d, h, w).to(dose_bev.dtype)
        dose_n = dose / dose.detach().amax(
            dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
        fluence_n = fluence / fluence.detach().amax(
            dim=(-3, -2, -1), keepdim=True).clamp_min(1e-12)
        path = (torch.cumsum(rho.clamp_min(0.0), dim=1)
                * self.depth_spacing_mm / max(self.path_scale_mm, 1e-6))
        dh, dw = self._lateral_density_differences(rho)
        x = torch.stack((
            dose_n.clamp(0.0, 2.0),
            fluence_n.clamp(0.0, 2.0),
            rho.clamp(0.0, 3.0) / 1.5 - 0.5,
            path.clamp(0.0, 2.0),
            dh.clamp(-1.5, 1.5) / 1.5,
            dw.clamp(-1.5, 1.5) / 1.5,
        ), dim=1)
        feat = self.features(x)
        # Remove external air only. Material type and interface strength are
        # explicit features, so the network can also correct lung/tissue and
        # tissue/bone boundaries instead of being hard-gated to lung.
        body_gate = torch.sigmoid((rho - 0.02) * 40.0)
        fraction = (self.max_fraction * torch.tanh(self.fraction_head(feat)[:, 0])
                    * body_gate)
        mixture = torch.softmax(self.mixture_head(feat), dim=1)
        self.last_fraction = fraction
        self.last_mixture = mixture

        correction = torch.zeros_like(dose_bev)
        for i, sigma_mm in enumerate(self.sigmas_mm):
            source = (dose * fraction * mixture[:, i]).reshape(b, g, d, h, w)
            spread = _gaussian_blur_lateral(source, sigma_mm / float(res_lat_mm))
            correction = correction + spread - source
        return (dose_bev + correction).clamp_min(0.0)
