#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Full4DenoiserConfig:
    # Descriptive name for new configs. Older checkpoints keep their original
    # name in the serialized config and remain loadable because unknown config
    # fields are ignored by the loader below.
    model_name: str = "full4_denoiser_logres_v1"
    in_ch: int = 4
    base: int = 25
    embed_dim: int = 64
    residual_scale_init: float = 0.5
    min_variance: float = 1.0e-8
    max_variance: float = 0.25
    a_scale: float = 1.0e-3
    b_scale: float = 1.0e-4
    local_log_variance_scale: float = 1.38629436112
    local_ch_min: int = 8
    local_channels: int = 192
    local_downsample: int = 4
    local_target_downsample: int = 4
    relative_grid_size: int = 64


class FiLM(nn.Module):
    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        hidden = max(int(embed_dim), int(channels))
        self.to_scale_shift = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(token).chunk(2, dim=1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


def _box5(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(F.pad(x.clamp(0.0, 1.0), (2, 2, 2, 2), mode="reflect"), 5, stride=1)


def _global_variance(a: torch.Tensor, b: torch.Tensor, signal: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    if a.ndim == 1:
        a = a[None, :, None, None]
    elif a.ndim == 2:
        a = a[:, :, None, None]
    if b.ndim == 1:
        b = b[None, :, None, None]
    elif b.ndim == 2:
        b = b[:, :, None, None]
    return (a * _box5(signal) + b).clamp(float(v_min), float(v_max))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _NoiseEncoder(nn.Module):
    def __init__(self, config: Full4DenoiserConfig):
        super().__init__()
        if (config.local_downsample, config.local_target_downsample, config.local_channels) != (4, 4, 192):
            raise ValueError("The package loader only supports the trained Full4 configuration.")
        state_ch = max(int(config.local_ch_min), int(config.base) // 2)
        self.config = config
        self.state_stem = nn.Sequential(
            nn.Conv2d(config.in_ch, state_ch, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(state_ch, state_ch * 2, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(state_ch * 2, config.embed_dim, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(nn.Linear(config.embed_dim, config.embed_dim), nn.SiLU(inplace=True))
        self.state_head = nn.Linear(config.embed_dim, config.embed_dim)
        self.a_head = nn.Linear(config.embed_dim, config.in_ch)
        self.b_head = nn.Linear(config.embed_dim, config.in_ch)
        self.local = nn.Sequential(
            nn.Conv2d(config.in_ch, config.local_channels, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.local_channels, config.local_channels, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.local_channels, config.in_ch, 3, padding=1),
        )
        self.relative_prior = nn.Parameter(torch.zeros(1, config.in_ch, config.relative_grid_size, config.relative_grid_size))
        self._init_heads()

    def _init_heads(self) -> None:
        for head in (self.state_head, self.a_head, self.b_head):
            nn.init.zeros_(head.weight)
        nn.init.zeros_(self.state_head.bias)
        nn.init.constant_(self.a_head.bias, -2.0)
        nn.init.constant_(self.b_head.bias, -4.0)
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, noisy_raw: torch.Tensor, relative_coords: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = noisy_raw.float()
        token_base = self.global_mlp(self.state_stem(x).mean(dim=(2, 3)))
        state_token = self.state_head(token_base)
        a_hat = F.softplus(self.a_head(token_base)) * float(self.config.a_scale)
        b_hat = F.softplus(self.b_head(token_base)) * float(self.config.b_scale)
        base_var = _global_variance(a_hat, b_hat, x.detach(), self.config.min_variance, self.config.max_variance)

        local_logits = self.local(x)
        h, w = local_logits.shape[-2:]
        if relative_coords is None:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=x.device, dtype=x.dtype),
                torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij",
            )
            grid = torch.stack(((xx + 0.5) / w * 2 - 1, (yy + 0.5) / h * 2 - 1), dim=-1)[None]
            grid = grid.expand(x.shape[0], -1, -1, -1)
        else:
            grid = relative_coords.to(device=x.device, dtype=x.dtype)
            if grid.shape != (x.shape[0], h, w, 2):
                raise ValueError(f"relative_coords must be [B,{h},{w},2], got {tuple(grid.shape)}")
        local_logits = local_logits + F.grid_sample(
            self.relative_prior.expand(x.shape[0], -1, -1, -1), grid, mode="bilinear", align_corners=False
        )
        residual_log_low = torch.tanh(local_logits) * float(self.config.local_log_variance_scale)
        residual_log = F.interpolate(residual_log_low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        noise_variance = (base_var * torch.exp(residual_log)).clamp(self.config.min_variance, self.config.max_variance)
        return {
            "noise_variance": noise_variance,
            "base_noise_variance": base_var,
            "state_token": state_token,
            "a_hat": a_hat,
            "b_hat": b_hat,
            "residual_log_scale": residual_log_low,
            "local_variance_ratio": torch.exp(residual_log_low),
            "residual_variance": torch.exp(residual_log_low),
        }


class _CleanUNet(nn.Module):
    def __init__(self, config: Full4DenoiserConfig):
        super().__init__()
        base = int(config.base)
        self.config = config
        self.enc1 = _ConvBlock(config.in_ch * 2, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.bottleneck_film = FiLM(config.embed_dim, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.clean_head = nn.Conv2d(base, config.in_ch, 3, padding=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))
        nn.init.zeros_(self.clean_head.weight)
        nn.init.zeros_(self.clean_head.bias)

    def forward(self, noisy_raw: torch.Tensor, state_token: torch.Tensor, noise_variance: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([noisy_raw, noise_variance.clamp_min(self.config.min_variance).sqrt().to(noisy_raw)], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.bottleneck_film(self.enc3(self.pool2(e2)), state_token)
        d2 = self.up2(e3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        residual = torch.tanh(self.clean_head(self.dec1(torch.cat([d1, e1], dim=1)))) * self.residual_scale.abs().clamp(0.01, 0.5)
        return (noisy_raw - residual).clamp(0.0, 1.0)


class Full4Denoiser(nn.Module):
    def __init__(self, config: Full4DenoiserConfig):
        super().__init__()
        self.config = config
        self.noise_encoder = _NoiseEncoder(config)
        self.denoiser = _CleanUNet(config)

    def forward(self, noisy_raw: torch.Tensor, relative_coords: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        state = self.noise_encoder(noisy_raw, relative_coords)
        return {
            "denoised_raw": self.denoiser(noisy_raw, state["state_token"], state["noise_variance"]),
            **state,
        }


class Full4DenoiserAdapter(nn.Module):
    """Expose the small dictionary interface expected by the ISP."""

    def __init__(self, core: Full4Denoiser, min_variance: float):
        super().__init__()
        self.core = core
        self.min_variance = float(min_variance)

    def set_trainable(self, trainable: str = "none") -> None:
        enabled = str(trainable).lower() not in {"none", "false", "0", "off"}
        for parameter in self.parameters():
            parameter.requires_grad_(enabled)

    def forward(self, raw4: torch.Tensor, _condition: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = self.core(raw4)
        clean = out["denoised_raw"].clamp(0, 1)
        variance = out["noise_variance"].clamp_min(self.min_variance)
        return {
            "clean": clean,
            "variance": variance,
            "sigma": variance.sqrt(),
            "log_var": torch.log(variance),
            "phys_var": variance,
            "relative_noise": torch.zeros_like(clean),
            "reliability": torch.ones_like(clean),
            "local_signal": clean.detach().clamp_min(1.0 / 4095.0),
            "residual": clean - raw4,
            "state_token": out["state_token"],
            "a_hat": out["a_hat"],
            "b_hat": out["b_hat"],
            "residual_variance": out["residual_variance"],
            "global_variance": out["base_noise_variance"],
            "residual_log_scale": out["residual_log_scale"],
            "local_variance_ratio": out["local_variance_ratio"],
        }


def _load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_full4_denoiser_from_checkpoint(path: str | Path, device: str = "cpu", trainable: str = "none"):
    checkpoint = _load_checkpoint(path)
    if checkpoint.get("model_name") != "stable_noise_adaptive_denoiser_logres_v1":
        raise ValueError("The package requires a Full4 stable log-residual denoiser checkpoint.")
    raw_config = dict(checkpoint.get("config", {}).get("model", {}))
    allowed = {field.name for field in fields(Full4DenoiserConfig)}
    values = {key: value for key, value in raw_config.items() if key in allowed}
    config = Full4DenoiserConfig(**values)
    expected = {
        "in_ch": 4,
        "base": 25,
        "embed_dim": 64,
        "local_ch_min": 8,
        "local_channels": 192,
        "local_downsample": 4,
        "local_target_downsample": 4,
    }
    mismatched = {key: (getattr(config, key), value) for key, value in expected.items() if getattr(config, key) != value}
    if mismatched:
        raise ValueError(f"Checkpoint is not the bundled Full4 configuration: {mismatched}")
    core = Full4Denoiser(config)
    state = checkpoint.get("ema_state") or checkpoint.get("model_state")
    if state is None:
        raise ValueError(f"Checkpoint has no model state: {path}")
    core.load_state_dict(state, strict=True)
    model = Full4DenoiserAdapter(core, min_variance=config.min_variance).to(device)
    model.set_trainable(trainable)
    return model, {
        "pattern": str(checkpoint.get("raw_spec", {}).get("pattern", "GBRG")),
        "model_type": config.model_name,
        "step": checkpoint.get("step"),
        "stage": checkpoint.get("stage"),
        "best_val_psnr": checkpoint.get("best_val_psnr"),
        "raw_spec": dict(checkpoint.get("raw_spec", {})),
    }


def make_condition_tensor(exposure_us, exp_min_log2, exp_max_log2, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(exposure_us, torch.Tensor):
        exp = exposure_us.detach().float().reshape(-1)
    else:
        exp = torch.as_tensor(exposure_us, dtype=torch.float32).reshape(-1)
    if device is not None:
        exp = exp.to(device=device)
    cond = 2.0 * (torch.log2(exp.clamp_min(1.0e-12)) - float(exp_min_log2)) / max(float(exp_max_log2) - float(exp_min_log2), 1.0e-12) - 1.0
    return cond.clamp(-1.5, 1.5).to(dtype=dtype).unsqueeze(1)


def pack_bayer_tensor(raw: torch.Tensor, pattern: str = "GBRG") -> torch.Tensor:
    if raw.ndim == 2:
        raw = raw[None, None]
    elif raw.ndim == 3:
        raw = raw[None]
    if raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError(f"Expected RAW [B,1,H,W], got {tuple(raw.shape)}")
    x00, x01 = raw[:, :, 0::2, 0::2], raw[:, :, 0::2, 1::2]
    x10, x11 = raw[:, :, 1::2, 0::2], raw[:, :, 1::2, 1::2]
    pattern = pattern.upper()
    if pattern == "GBRG":
        channels = (x10, x00, x11, x01)
    elif pattern == "RGGB":
        channels = (x00, x01, x10, x11)
    else:
        raise ValueError(f"The reviewer package supports GBRG (or RGGB for utility use), got {pattern}")
    return torch.cat(channels, dim=1)


def unpack_bayer_tensor(x4: torch.Tensor, pattern: str = "GBRG") -> torch.Tensor:
    if x4.ndim != 4 or x4.shape[1] != 4:
        raise ValueError(f"Expected packed RAW [B,4,h,w], got {tuple(x4.shape)}")
    b, _, h, w = x4.shape
    out = x4.new_zeros((b, 1, h * 2, w * 2))
    r, g1, g2, blue = x4[:, 0:1], x4[:, 1:2], x4[:, 2:3], x4[:, 3:4]
    pattern = pattern.upper()
    if pattern == "GBRG":
        out[:, :, 0::2, 0::2], out[:, :, 0::2, 1::2] = g1, blue
        out[:, :, 1::2, 0::2], out[:, :, 1::2, 1::2] = r, g2
    elif pattern == "RGGB":
        out[:, :, 0::2, 0::2], out[:, :, 0::2, 1::2] = r, g1
        out[:, :, 1::2, 0::2], out[:, :, 1::2, 1::2] = g2, blue
    else:
        raise ValueError(f"The reviewer package supports GBRG (or RGGB for utility use), got {pattern}")
    return out


def complete_denoiser_output(out: dict[str, torch.Tensor], min_variance: float = 1.0e-8) -> dict[str, torch.Tensor]:
    clean = out["clean"]
    variance = out.setdefault("variance", torch.full_like(clean, float(min_variance)))
    out.setdefault("log_var", torch.log(variance.clamp_min(float(min_variance))))
    out.setdefault("sigma", variance.clamp_min(0).sqrt())
    out.setdefault("phys_var", variance.clamp_min(float(min_variance)))
    out.setdefault("relative_noise", torch.zeros_like(clean))
    out.setdefault("reliability", torch.ones_like(clean))
    out.setdefault("local_signal", clean.detach().clamp_min(1.0 / 4095.0))
    return out


__all__ = [
    "Full4DenoiserConfig",
    "Full4Denoiser",
    "build_full4_denoiser_from_checkpoint",
    "complete_denoiser_output",
    "make_condition_tensor",
    "pack_bayer_tensor",
    "unpack_bayer_tensor",
]
