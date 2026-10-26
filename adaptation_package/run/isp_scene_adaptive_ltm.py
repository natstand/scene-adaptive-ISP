#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import demosaic_malvar2004, gaussian_blur
from .full4_denoiser import pack_bayer_tensor, unpack_bayer_tensor
from .isp import ResidualISPV4, range_stats


class ResidualISPV4SceneAdaptiveLTM(ResidualISPV4):
    """ResidualISPV4 with scene-adaptive local tone mapping.

    The model learns only five scene-level LTM parameters. During rendering, the
    local EV map is generated deterministically from the current frame's
    low-resolution smoothed luminance.
    """

    def __init__(
        self,
        *args,
        ltm_guidance_max_side: int = 256,
        disable_ltm_dark_boost: bool = False,
        ltm_black_guard_min: float = 0.05,
        ltm_black_threshold: float = 0.03,
        ltm_black_softness: float = 0.02,
        ltm_use_rgb_black_guard: bool = False,
        ltm_use_raw_black_guard: bool = True,
        ltm_raw_black_threshold: float = 0.015,
        ltm_raw_black_softness: float = 0.02,
        disable_scene_adaptive_ltm: bool = False,
        disable_ltm_highlight_compression: bool = False,
        ltm_fixed_thresholds: bool = False,
        ltm_fixed_blend_strength: float = -1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Remove the fixed spatial LTM map from the base class.
        if hasattr(self, "ltm_maps"):
            delattr(self, "ltm_maps")

        # Scene-adaptive LTM scene-level parameters:
        # dark EV, highlight EV, dark threshold, highlight threshold, blend.
        self.du_ltm = nn.Parameter(
            torch.tensor([-2.0, -2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        )
        self.register_buffer("ltm_transition_width", torch.tensor(0.08, dtype=torch.float32))
        self.register_buffer("ltm_black_threshold", torch.tensor(float(ltm_black_threshold), dtype=torch.float32))
        self.register_buffer("ltm_black_softness", torch.tensor(float(ltm_black_softness), dtype=torch.float32))
        self.register_buffer("ltm_raw_black_threshold", torch.tensor(float(ltm_raw_black_threshold), dtype=torch.float32))
        self.register_buffer("ltm_raw_black_softness", torch.tensor(float(ltm_raw_black_softness), dtype=torch.float32))
        self.ltm_guidance_max_side = int(ltm_guidance_max_side)
        self.disable_ltm_dark_boost = bool(disable_ltm_dark_boost)
        self.ltm_black_guard_min = float(ltm_black_guard_min)
        self.ltm_use_rgb_black_guard = bool(ltm_use_rgb_black_guard)
        self.ltm_use_raw_black_guard = bool(ltm_use_raw_black_guard)
        # Explicit ablation controls.  Defaults preserve the paper model.
        self.disable_scene_adaptive_ltm = bool(disable_scene_adaptive_ltm)
        self.disable_ltm_highlight_compression = bool(disable_ltm_highlight_compression)
        self.ltm_fixed_thresholds = bool(ltm_fixed_thresholds)
        self.ltm_fixed_blend_strength = float(ltm_fixed_blend_strength)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_ltm_key = prefix + "ltm_maps"
        du_ltm_key = prefix + "du_ltm"
        if legacy_ltm_key in state_dict:
            state_dict.pop(legacy_ltm_key)
        if du_ltm_key not in state_dict:
            state_dict[du_ltm_key] = self.du_ltm.detach().clone()
        else:
            loaded = state_dict[du_ltm_key]
            if loaded.numel() == 6 and self.du_ltm.numel() == 5:
                state_dict[du_ltm_key] = loaded.reshape(-1)[:5].reshape_as(self.du_ltm)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def scene_adaptive_ltm_params(self) -> dict[str, torch.Tensor]:
        p = self.du_ltm
        dark_ev = 1.25 * torch.sigmoid(p[0])
        if self.disable_ltm_dark_boost:
            dark_ev = p.new_zeros(())
        dark_threshold = 0.05 + 0.30 * torch.sigmoid(p[2])
        highlight_threshold = 0.60 + 0.35 * torch.sigmoid(p[3])
        # Fixed values equal the documented default initialization.
        if self.ltm_fixed_thresholds:
            dark_threshold = p.new_tensor(0.10)
            highlight_threshold = p.new_tensor(0.88)
        blend_strength = torch.sigmoid(p[4])
        if self.ltm_fixed_blend_strength >= 0.0:
            blend_strength = p.new_tensor(self.ltm_fixed_blend_strength).clamp(0.0, 1.0)
        return {
            "dark_ev": dark_ev,
            "highlight_ev": p.new_zeros(()) if self.disable_ltm_highlight_compression else 1.00 * torch.sigmoid(p[1]),
            "dark_threshold": dark_threshold,
            "highlight_threshold": highlight_threshold,
            "blend_strength": blend_strength,
        }

    @staticmethod
    def rgb_luminance(rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
        w = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        return (rgb * w).sum(dim=1, keepdim=True)

    @staticmethod
    def _ltm_guidance_size(height: int, width: int, max_side: int = 256) -> tuple[int, int]:
        height = int(height)
        width = int(width)
        max_side = int(max_side)
        if max(height, width) <= max_side:
            return height, width
        scale = max_side / float(max(height, width))
        out_h = max(16, int(round(height * scale)))
        out_w = max(16, int(round(width * scale)))
        return out_h, out_w

    def make_scene_adaptive_ltm_map(
        self,
        rgb_pre_gtm: torch.Tensor,
        raw_denoised_full: torch.Tensor | None = None,
        max_guidance_side: int = 256,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if rgb_pre_gtm.ndim != 4 or rgb_pre_gtm.shape[1] != 3:
            raise ValueError(
                f"rgb_pre_gtm must have shape [B,3,H,W], got {tuple(rgb_pre_gtm.shape)}"
            )

        _, _, height, width = rgb_pre_gtm.shape
        params = self.scene_adaptive_ltm_params()

        # Routing guidance is detached so upstream ISP parameters are not pushed
        # to manipulate only the LTM masks.
        y = self.rgb_luminance(rgb_pre_gtm.detach().clamp(0.0, 1.0)).clamp(0.0, 1.0)
        guide_h, guide_w = self._ltm_guidance_size(
            height,
            width,
            max_side=max_guidance_side,
        )
        y_small = F.interpolate(y, size=(guide_h, guide_w), mode="area")
        y_guide = gaussian_blur(y_small, kernel_size=9, sigma=2.0).clamp(0.0, 1.0)

        transition_width = self.ltm_transition_width.to(
            dtype=y_guide.dtype,
            device=y_guide.device,
        ).clamp_min(1e-4)
        dark_weight = torch.sigmoid((params["dark_threshold"] - y_guide) / transition_width)
        highlight_weight = torch.sigmoid(
            (y_guide - params["highlight_threshold"]) / transition_width
        )

        black_threshold = self.ltm_black_threshold.to(
            dtype=y_guide.dtype,
            device=y_guide.device,
        )
        black_softness = self.ltm_black_softness.to(
            dtype=y_guide.dtype,
            device=y_guide.device,
        ).clamp_min(1e-4)
        guard_min = min(max(float(self.ltm_black_guard_min), 0.0), 1.0)
        rgb_black_guard = guard_min + (1.0 - guard_min) * torch.sigmoid(
            (y_guide - black_threshold) / black_softness
        )
        black_guard = rgb_black_guard if self.ltm_use_rgb_black_guard else torch.ones_like(rgb_black_guard)

        raw_guard = torch.ones_like(black_guard)
        raw_guide = torch.zeros_like(y_guide)
        if self.ltm_use_raw_black_guard and raw_denoised_full is not None:
            if raw_denoised_full.ndim != 4 or raw_denoised_full.shape[1] != 1:
                raise ValueError(
                    "raw_denoised_full must have shape [B,1,H,W], "
                    f"got {tuple(raw_denoised_full.shape)}"
                )
            raw4 = pack_bayer_tensor(raw_denoised_full.detach().clamp(0.0, 1.0), pattern=self.pattern)
            raw_green = 0.5 * (raw4[:, 1:2] + raw4[:, 2:3])
            raw_small = F.interpolate(
                raw_green.to(dtype=y_guide.dtype, device=y_guide.device),
                size=(guide_h, guide_w),
                mode="area",
            )
            raw_guide = gaussian_blur(raw_small, kernel_size=9, sigma=2.0).clamp(0.0, 1.0)
            raw_threshold = self.ltm_raw_black_threshold.to(
                dtype=y_guide.dtype,
                device=y_guide.device,
            )
            raw_softness = self.ltm_raw_black_softness.to(
                dtype=y_guide.dtype,
                device=y_guide.device,
            ).clamp_min(1e-4)
            raw_guard = torch.sigmoid((raw_guide - raw_threshold) / raw_softness)

        local_ev_small = (
            params["dark_ev"] * dark_weight * black_guard * raw_guard
            - params["highlight_ev"] * highlight_weight
        )

        max_local_ev = 1.25
        local_ev_small = max_local_ev * torch.tanh(local_ev_small / max_local_ev)
        local_ev = F.interpolate(
            local_ev_small,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        debug = {
            "ltm_y_guide": y_guide,
            "ltm_dark_weight": dark_weight,
            "ltm_highlight_weight": highlight_weight,
            "ltm_black_guard": black_guard,
            "ltm_rgb_black_guard": rgb_black_guard,
            "ltm_raw_guide": raw_guide,
            "ltm_raw_black_guard": raw_guard,
            "ltm_local_ev_small": local_ev_small,
            "ltm_local_ev": local_ev,
        }
        return local_ev, debug

    def apply_scene_adaptive_ltm(
        self,
        rgb_pre_gtm: torch.Tensor,
        rgb_gtm: torch.Tensor,
        raw_denoised_full: torch.Tensor | None = None,
        return_debug: bool = False,
    ):
        if rgb_pre_gtm.shape != rgb_gtm.shape:
            raise ValueError(
                "rgb_pre_gtm and rgb_gtm must have identical shapes, "
                f"got {tuple(rgb_pre_gtm.shape)} and {tuple(rgb_gtm.shape)}"
            )

        if self.disable_scene_adaptive_ltm:
            debug = {"ltm_disabled": torch.ones((), device=rgb_gtm.device, dtype=rgb_gtm.dtype)}
            return (rgb_gtm, debug) if return_debug else rgb_gtm
        params = self.scene_adaptive_ltm_params()
        local_ev, debug = self.make_scene_adaptive_ltm_map(
            rgb_pre_gtm=rgb_pre_gtm,
            raw_denoised_full=raw_denoised_full,
            max_guidance_side=self.ltm_guidance_max_side,
        )

        y_global = self.rgb_luminance(rgb_gtm.clamp(0.0, 1.0)).clamp_min(1e-4)
        y_local = (y_global * torch.pow(local_ev.new_tensor(2.0), local_ev)).clamp(1e-4, 1.0)
        local_scale = (y_local / y_global).clamp(0.5, 2.0)
        rgb_local = (rgb_gtm * local_scale).clamp(0.0, 1.0)

        blend = params["blend_strength"].view(1, 1, 1, 1)
        rgb_out = ((1.0 - blend) * rgb_gtm + blend * rgb_local).clamp(0.0, 1.0)

        debug.update(
            {
                "ltm_y_global": y_global,
                "ltm_y_local": y_local,
                "ltm_local_scale": local_scale,
                "ltm_blend_strength": blend,
            }
        )
        if return_debug:
            return rgb_out, debug
        return rgb_out

    def make_guidance(self, rgb_gain, rgb_gtm):
        local_ev, _ = self.make_scene_adaptive_ltm_map(
            rgb_pre_gtm=rgb_gain,
            max_guidance_side=self.ltm_guidance_max_side,
        )
        return local_ev.abs().clamp(0.0, 1.0)

    def apply_ltm(self, rgb_gain, rgb_gtm, raw_reliability=None, return_debug=False):
        rgb_out, debug = self.apply_scene_adaptive_ltm(
            rgb_pre_gtm=rgb_gain,
            rgb_gtm=rgb_gtm,
            return_debug=True,
        )
        if return_debug:
            return rgb_out, debug
        return rgb_out

    def render(self, raw, exp_t):
        cache = self._cached_frontend()
        if cache is not None:
            raw_dn_full = cache["raw_denoised_full"]
            rgb = cache["linear_rgb"]
        else:
            raw_packed = pack_bayer_tensor(raw, pattern=self.pattern)
            den = self._run_raw_denoiser(raw_packed, exp_t)
            raw_dn_packed = den["clean"] if self.use_denoiser_clean_prior else raw_packed
            raw_dn_full = unpack_bayer_tensor(raw_dn_packed, pattern=self.pattern).clamp(0, 1)
            rgb = demosaic_malvar2004(raw_dn_full, pattern=self.pattern)

        rgb = rgb * self.exposure() * self.wb_gain()
        rgb = self.apply_ccm(rgb).clamp_min(0.0)
        rgb_pre_gtm = rgb
        rgb_gtm = self.apply_gtm(rgb_pre_gtm)
        rgb = self.apply_scene_adaptive_ltm(
            rgb_pre_gtm=rgb_pre_gtm,
            rgb_gtm=rgb_gtm,
            raw_denoised_full=raw_dn_full,
        )
        if self.lut is not None:
            rgb = self.lut(rgb).clamp(0.0, 1.0)
        rgb = self.apply_saturation(rgb).clamp(0.0, 1.0)
        return rgb

    def forward_debug(self, raw, exp_t):
        out = {}
        cache = self._cached_frontend()
        if cache is not None:
            out["raw_denoised_full"] = cache["raw_denoised_full"]
            out["raw_variance_full"] = cache["raw_variance_full"]
            out["raw_confidence"] = cache["raw_confidence"]
            rgb_linear = cache["linear_rgb"]
        else:
            raw_packed = pack_bayer_tensor(raw, pattern=self.pattern)
            den = self._run_raw_denoiser(raw_packed, exp_t)
            raw_dn = (den["clean"] if self.use_denoiser_clean_prior else raw_packed).clamp(0, 1)
            out["raw_denoised_full"] = unpack_bayer_tensor(raw_dn, pattern=self.pattern)
            out["raw_variance_full"] = unpack_bayer_tensor(den["variance"].clamp_min(0), pattern=self.pattern)
            out["raw_confidence"] = unpack_bayer_tensor(
                torch.ones_like(raw_packed),
                pattern=self.pattern,
            )
            rgb_linear = demosaic_malvar2004(out["raw_denoised_full"], pattern=self.pattern)

        out["linear_rgb"] = rgb_linear
        out["after_exposure"] = rgb_linear * self.exposure()
        out["after_wb"] = out["after_exposure"] * self.wb_gain()
        out["after_ccm"] = self.apply_ccm(out["after_wb"]).clamp_min(0.0)
        rgb_gtm = self.apply_gtm(out["after_ccm"])
        out["after_gtm"] = rgb_gtm
        rgb_ltm, ltm_debug = self.apply_scene_adaptive_ltm(
            rgb_pre_gtm=out["after_ccm"],
            rgb_gtm=rgb_gtm,
            raw_denoised_full=out["raw_denoised_full"],
            return_debug=True,
        )
        out["after_ltm"] = rgb_ltm
        out["after_tone"] = rgb_ltm
        out.update(ltm_debug)
        out["before_lut"] = rgb_ltm.clamp(0.0, 1.0)
        out["after_lut"] = (
            self.lut(out["before_lut"]).clamp(0.0, 1.0)
            if self.lut is not None
            else out["before_lut"]
        )
        out["after_saturation"] = self.apply_saturation(out["after_lut"]).clamp(0.0, 1.0)
        out["final"] = out["after_saturation"]
        out["range_stats"] = {
            k: range_stats(out[k])
            for k in ("after_wb", "after_ccm", "after_tone", "after_saturation", "before_lut", "after_lut")
            if k in out
        }
        return out

    def ltm_regularization(
        self,
        strength_weight: float = 1e-3,
        activation_weight: float = 5e-4,
    ) -> torch.Tensor:
        params = self.scene_adaptive_ltm_params()
        loss_strength = params["dark_ev"].square() + params["highlight_ev"].square()
        loss_activation = params["blend_strength"].square()
        return float(strength_weight) * loss_strength + float(activation_weight) * loss_activation


__all__ = ["ResidualISPV4SceneAdaptiveLTM"]
