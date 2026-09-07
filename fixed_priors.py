"""Small frozen neural priors used by the compact photon dose corrector."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyFluenceCorrector(nn.Module):
    """Identity-initialised 2-D U-Net producing a bounded fluence log-gain."""

    def __init__(self, base_channels: int = 8,
                 max_log_gain: float = math.log(1.5)):
        super().__init__()
        b = int(base_channels)
        if b < 2:
            raise ValueError("base_channels must be >= 2")
        self.max_log_gain = float(max_log_gain)
        self.enc0 = ConvBlock2d(1, b)
        self.down1 = nn.Conv2d(b, 2 * b, 3, stride=2, padding=1, bias=False)
        self.enc1 = ConvBlock2d(2 * b, 2 * b)
        self.down2 = nn.Conv2d(2 * b, 4 * b, 3, stride=2, padding=1, bias=False)
        self.bottleneck = ConvBlock2d(4 * b, 4 * b)
        self.up1 = nn.Conv2d(4 * b, 2 * b, 1)
        self.dec1 = ConvBlock2d(4 * b, 2 * b)
        self.up0 = nn.Conv2d(2 * b, b, 1)
        self.dec0 = ConvBlock2d(2 * b, b)
        self.head = nn.Conv2d(b, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.last_log_gain: torch.Tensor | None = None

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(raw)
        e1 = self.enc1(self.down1(e0))
        z = self.bottleneck(self.down2(e1))
        z = F.interpolate(z, size=e1.shape[-2:], mode="bilinear",
                          align_corners=False)
        z = self.dec1(torch.cat((self.up1(z), e1), dim=1))
        z = F.interpolate(z, size=e0.shape[-2:], mode="bilinear",
                          align_corners=False)
        z = self.dec0(torch.cat((self.up0(z), e0), dim=1))
        log_gain = self.max_log_gain * torch.tanh(self.head(z))
        self.last_log_gain = log_gain
        corrected = raw * torch.exp(log_gain)
        return corrected - raw


def build_tiny_fluence(config: dict, state_dict: dict, device):
    max_gain = float(config.get("max_gain", 1.5))
    model = TinyFluenceCorrector(
        base_channels=int(config.get("base_channels", 8)),
        max_log_gain=math.log(max_gain),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.requires_grad_(False).to(device).to(torch.float32).eval()


def build_tiny_heterogeneity(config: dict, state_dict: dict, device):
    from lateral_scatter import TinyLateralHeterogeneityCorrection

    architecture = config.get("architecture")
    if architecture not in (None, "heterogeneity"):
        raise ValueError(f"expected heterogeneity checkpoint, got {architecture!r}")
    model = TinyLateralHeterogeneityCorrection(
        hidden_channels=int(config.get("hidden_channels", 4)),
        sigmas_mm=tuple(float(x) for x in config.get("sigmas_mm", (2, 5, 10))),
        max_fraction=float(config.get("max_fraction", 0.4)),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.requires_grad_(False).to(device).to(torch.float32).eval()


def build_embedded_priors(checkpoint: dict, device):
    """Reconstruct fixed priors embedded in a final corrector checkpoint."""
    cfg = checkpoint.get("model_config") or {}
    fluence_cfg = cfg.get("fixed_fluence_config")
    lateral_cfg = cfg.get("fixed_lateral_heterogeneity_config")
    fluence_state = checkpoint.get("fluence_model_state_dict")
    lateral_state = checkpoint.get("lateral_heterogeneity_model_state_dict")
    if bool(fluence_cfg) != bool(fluence_state):
        raise ValueError("fixed fluence config/state must either both exist or both be absent")
    if bool(lateral_cfg) != bool(lateral_state):
        raise ValueError("fixed heterogeneity config/state must both exist or be absent")
    fluence = (build_tiny_fluence(fluence_cfg, fluence_state, device)
               if fluence_cfg else None)
    lateral = (build_tiny_heterogeneity(lateral_cfg, lateral_state, device)
               if lateral_cfg else None)
    return fluence, lateral
