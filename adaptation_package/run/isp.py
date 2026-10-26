#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .common import *
from .common import _rgb_luminance

# ============================================================
# Init net and residual ISP
# ============================================================

def soft_clip01_ste(x: torch.Tensor) -> torch.Tensor:
    clipped = x.clamp(0.0, 1.0)
    return x + (clipped - x).detach()


class LUT3D(nn.Module):
    def __init__(self, size: int = 33):
        super().__init__()
        self.chroma_only = False
        self.lut_kind = "rgb"
        # Black-safe LUT defaults.  The table remains fully learnable above
        # the near-black transition, while pure/near black cannot acquire a
        # LUT-induced colour cast.
        self.black_anchor = True
        self.near_black_fade = True
        self.near_black_low = 0.030
        self.near_black_high = 0.150
        grid = torch.linspace(0.0, 1.0, int(size))
        r, g, b = torch.meshgrid(grid, grid, grid, indexing="ij")
        self.lut = nn.Parameter(torch.stack([r, g, b], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = soft_clip01_ste(x)
        B, _, H, W = x.shape
        grid = torch.cat([x[:, 2:3], x[:, 1:2], x[:, 0:1]], dim=1).permute(0, 2, 3, 1) * 2.0 - 1.0
        grid = grid.unsqueeze(1)
  
        table = self.lut
        if self.black_anchor:
            table = table.clone()
            table[0, 0, 0] = 0.0
        lut = table.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        out = F.grid_sample(lut.expand(B, -1, -1, -1, -1), grid, mode="bilinear", padding_mode="border", align_corners=True)
        out = soft_clip01_ste(out.squeeze(2))
        if self.near_black_fade:
            low = max(float(self.near_black_low), 0.0)
            high = max(float(self.near_black_high), low + 1e-6)
            y = _rgb_luminance(x).clamp(0.0, 1.0)
            t = ((y - low) / (high - low)).clamp(0.0, 1.0)
            weight = t * t * (3.0 - 2.0 * t)  # smoothstep
            out = x + weight * (out - x)
        return soft_clip01_ste(out)


def rgb_to_ycbcr_bt709(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    kr, kg, kb = 0.2126, 0.7152, 0.0722
    y = kr * r + kg * g + kb * b
    cb = 0.5 + (b - y) / (2.0 * (1.0 - kb))
    cr = 0.5 + (r - y) / (2.0 * (1.0 - kr))
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb_bt709(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2] - 0.5, ycbcr[:, 2:3] - 0.5
    kr, kg, kb = 0.2126, 0.7152, 0.0722
    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return torch.cat([r, g, b], dim=1)


class YCbCrChromaLUT2D(nn.Module):
    """2D LUT indexed only by Cb/Cr, predicting chroma deltas.

    Y is kept unchanged and is not used as a LUT coordinate. The learned table
    stores unconstrained delta-Cb/Cr logits and the forward path bounds their
    effect with tanh.
    """

    def __init__(self, size: int = 33, delta_scale: float = 0.25):
        super().__init__()
        self.chroma_only = True
        self.lut_kind = "ycbcr_chroma_2d"
        self.delta_scale = float(delta_scale)
        self.lut = nn.Parameter(torch.zeros(int(size), int(size), 2))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "lut"
        if key in state_dict and tuple(state_dict[key].shape) != tuple(self.lut.shape):
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = soft_clip01_ste(x)
        B, _, H, W = x.shape
        ycbcr = rgb_to_ycbcr_bt709(x)
        y = ycbcr[:, 0:1]
        cbcr = ycbcr[:, 1:3]
        grid = torch.cat([ycbcr[:, 2:3], ycbcr[:, 1:2]], dim=1).permute(0, 2, 3, 1) * 2.0 - 1.0
        lut = self.lut.permute(2, 0, 1).unsqueeze(0).contiguous()
        delta = F.grid_sample(lut.expand(B, -1, -1, -1), grid, mode="bilinear", padding_mode="border", align_corners=True)
        delta = torch.tanh(delta) * self.delta_scale
        ycbcr_out = torch.cat([y, (cbcr + delta).clamp(0.0, 1.0)], dim=1)
        return soft_clip01_ste(ycbcr_to_rgb_bt709(ycbcr_out))


class YCbCrChromaLUT3D(nn.Module):
    """Legacy 3D Y/Cb/Cr chroma-only LUT used for loading old checkpoints."""

    def __init__(self, size: int = 17, delta_scale: float = 0.25):
        super().__init__()
        self.chroma_only = True
        self.lut_kind = "ycbcr_chroma_3d"
        self.delta_scale = float(delta_scale)
        self.lut = nn.Parameter(torch.zeros(int(size), int(size), int(size), 2))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "lut"
        if key in state_dict and tuple(state_dict[key].shape) != tuple(self.lut.shape):
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = soft_clip01_ste(x)
        B, _, H, W = x.shape
        ycbcr = rgb_to_ycbcr_bt709(x)
        y = ycbcr[:, 0:1]
        cbcr = ycbcr[:, 1:3]
        grid = torch.cat([ycbcr[:, 2:3], ycbcr[:, 1:2], ycbcr[:, 0:1]], dim=1).permute(0, 2, 3, 1) * 2.0 - 1.0
        grid = grid.unsqueeze(1)
        lut = self.lut.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        delta = F.grid_sample(lut.expand(B, -1, -1, -1, -1), grid, mode="bilinear", padding_mode="border", align_corners=True)
        delta = torch.tanh(delta.squeeze(2)) * self.delta_scale
        ycbcr_out = torch.cat([y, (cbcr + delta).clamp(0.0, 1.0)], dim=1)
        return soft_clip01_ste(ycbcr_to_rgb_bt709(ycbcr_out))


def lut_smooth_loss(lut_param: torch.Tensor):
    losses = []
    for dim in range(max(0, lut_param.ndim - 1)):
        if lut_param.shape[dim] > 1:
            losses.append(torch.diff(lut_param, dim=dim).abs().mean())
    return sum(losses) if losses else lut_param.new_tensor(0.0)


def lut_monotonic_loss(lut_param: torch.Tensor):
    return F.relu(-(lut_param[1:] - lut_param[:-1])).mean() + F.relu(-(lut_param[:, 1:] - lut_param[:, :-1])).mean() + F.relu(-(lut_param[:, :, 1:] - lut_param[:, :, :-1])).mean()


class InitISPNet(nn.Module):
    """Small per-pair init ISP used only to create an alignment reference."""
    def __init__(self, ccm_clamp_abs: float = 2.0):
        super().__init__()
        self.ccm_clamp_abs = float(ccm_clamp_abs)
        self.exposure_ev_range = 2.0
        self.register_buffer("s0", torch.tensor(1.0))
        self.register_buffer("log2_s0", torch.tensor(0.0))
        self.register_buffer("wb0_rb", torch.tensor([1.0, 1.0]))
        self.register_buffer("ccm0", torch.eye(3))
        self.du_s = nn.Parameter(torch.zeros(1))
        self.du_wb = nn.Parameter(torch.zeros(2))
        self.du_ccm = nn.Parameter(torch.zeros(3, 3))
        self.du_gamma = nn.Parameter(torch.zeros(1))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        log_key = prefix + "log2_s0"
        s0_key = prefix + "s0"
        du_key = prefix + "du_s"
        if log_key not in state_dict and s0_key in state_dict:
            s0 = state_dict[s0_key].detach().float()
            du = state_dict.get(du_key, torch.zeros_like(s0)).detach().float()
            legacy_gain = (s0 + torch.tanh(du) * 0.8).clamp(0.05, 40.0)
            state_dict[log_key] = torch.log2(legacy_gain).to(state_dict[s0_key])
            state_dict[s0_key] = legacy_gain.to(state_dict[s0_key])
            if du_key in state_dict:
                state_dict[du_key] = torch.zeros_like(state_dict[du_key])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def set_base_exposure(self, gain):
        gain_t = torch.as_tensor(gain, device=self.log2_s0.device, dtype=self.log2_s0.dtype).detach().clamp(0.05, 40.0)
        gain_t = gain_t.reshape_as(self.log2_s0)
        self.log2_s0.copy_(torch.log2(gain_t))
        self.s0.copy_(gain_t.to(self.s0))

    @torch.no_grad()
    def set_effective_exposure(self, gain):
        self.set_base_exposure(gain)
        self.du_s.zero_()

    def exposure(self):
        ev = self.log2_s0.to(self.du_s) + float(self.exposure_ev_range) * torch.tanh(self.du_s)
        gain = torch.pow(self.du_s.new_tensor(2.0), ev)
        return gain.clamp(0.05, 40.0).view(1, 1, 1, 1)

    def wb_gain(self):
        wb = (self.wb0_rb + torch.tanh(self.du_wb) * 0.8).clamp(0.2, 4.0)
        return torch.stack([wb[0], torch.ones_like(wb[0]), wb[1]]).view(1, 3, 1, 1)

    def ccm(self):
        M = (self.ccm0 + torch.tanh(self.du_ccm) * 0.20).clamp(-self.ccm_clamp_abs, self.ccm_clamp_abs)
        return M / M.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def apply_ccm(self, rgb):
        B, C, H, W = rgb.shape
        y = rgb.permute(0, 2, 3, 1).reshape(-1, 3) @ self.ccm().to(rgb).T
        return y.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def forward(self, linear_rgb):
        x = linear_rgb * self.exposure() * self.wb_gain()
        x = self.apply_ccm(x).clamp_min(0.0)
        x = linear_to_srgb(x.clamp(0.0, 1.0))
        gamma = (1.0 + torch.tanh(self.du_gamma) * 0.45).clamp(0.55, 1.45)
        return x.clamp(1e-6, 1.0).pow(1.0 / gamma.view(1, 1, 1, 1)).clamp(0.0, 1.0)


def fit_rgb_to_hw_letterbox(rgb: torch.Tensor, out_h: int, out_w: int, pad_value: float = 0.0) -> torch.Tensor:
    _, _, h, w = rgb.shape
    scale = min(out_h / max(h, 1), out_w / max(w, 1))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    x = F.interpolate(rgb, size=(nh, nw), mode="bilinear", align_corners=False)
    top = max(0, (out_h - nh) // 2)
    bottom = max(0, out_h - nh - top)
    left = max(0, (out_w - nw) // 2)
    right = max(0, out_w - nw - left)
    return F.pad(x, (left, right, top, bottom), mode="constant", value=float(pad_value)).clamp(0, 1)


def _masked_channel_ratio(num: torch.Tensor, den: torch.Tensor, sel: torch.Tensor, stat: str) -> torch.Tensor:
    stat = str(stat).lower()
    if stat == "mean":
        return num[sel].mean() / (den[sel].mean() + 1e-6)
    if stat == "median":
        return num[sel].median() / (den[sel].median() + 1e-6)
    raise ValueError(f"Unsupported init_wb_stat: {stat}")


def _phone_neutral_selector(phone_srgb: torch.Tensor, base_sel: torch.Tensor, min_count: int) -> torch.Tensor:
    phone_lab = srgb_to_lab_fast(phone_srgb.clamp(0, 1))
    chroma = torch.sqrt(phone_lab[:, 1:2].pow(2) + phone_lab[:, 2:3].pow(2))
    y = _rgb_luminance(srgb_to_linear(phone_srgb.clamp(0, 1)))
    neutral = base_sel & (chroma < 10.0) & (y > 0.05) & (y < 0.85)
    if int(neutral.detach().sum().item()) < int(min_count):
        neutral = base_sel & (chroma < 16.0) & (y > 0.03) & (y < 0.90)
    if int(neutral.detach().sum().item()) < int(min_count):
        neutral = base_sel
    return neutral


def init_from_unaligned_phone(
    init_net: InitISPNet,
    src_lin_full: torch.Tensor,
    phone_rgb: torch.Tensor,
    wb_mode: str = "ratio",
    wb_stat: str = "median",
    wb_min: float = 0.3,
    wb_max: float = 3.0,
):
    with torch.no_grad():
        phone_fit = fit_rgb_to_hw_letterbox(phone_rgb, src_lin_full.shape[-2], src_lin_full.shape[-1])
        src_small = F.interpolate(src_lin_full, size=(256, 256), mode="area")
        phone_small = F.interpolate(phone_fit, size=(256, 256), mode="area")
        phone_lin = srgb_to_linear(phone_small)
        y_s = (src_small * src_small.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        y_t = (phone_lin * phone_lin.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        sel = (y_s > 0.005) & (y_t > 0.02) & (y_t < 0.9)
        if sel.sum() < 512:
            sel = y_s > 0.002
        s0 = (y_t[sel].mean() / (y_s[sel].mean() + 1e-6)).clamp(0.05, 40.0)
        init_net.set_base_exposure(s0)
        Rs, Gs, Bs = src_small[:, 0:1], src_small[:, 1:2], src_small[:, 2:3]
        Rt, Gt, Bt = phone_lin[:, 0:1], phone_lin[:, 1:2], phone_lin[:, 2:3]
        phone_rg = _masked_channel_ratio(Rt, Gt, sel, wb_stat)
        phone_bg = _masked_channel_ratio(Bt, Gt, sel, wb_stat)
        if str(wb_mode).lower() == "phone":
            phone_sel = _phone_neutral_selector(phone_small, sel, min_count=512)
            wr = _masked_channel_ratio(phone_small[:, 0:1], phone_small[:, 1:2], phone_sel, wb_stat).clamp(float(wb_min), float(wb_max))
            wb = _masked_channel_ratio(phone_small[:, 2:3], phone_small[:, 1:2], phone_sel, wb_stat).clamp(float(wb_min), float(wb_max))
        elif str(wb_mode).lower() == "ratio":
            wr = (phone_rg / (_masked_channel_ratio(Rs, Gs, sel, wb_stat) + 1e-6)).clamp(float(wb_min), float(wb_max))
            wb = (phone_bg / (_masked_channel_ratio(Bs, Gs, sel, wb_stat) + 1e-6)).clamp(float(wb_min), float(wb_max))
        else:
            raise ValueError(f"Unsupported init_wb_mode: {wb_mode}")
        init_net.wb0_rb.copy_(torch.stack([wr, wb]).to(init_net.wb0_rb))


def optimize_init_net(init_net: InitISPNet, src_lin_full: torch.Tensor, phone_rgb: torch.Tensor, steps=80, lr=2e-3, down=256):
    if steps <= 0:
        return
    phone_fit = fit_rgb_to_hw_letterbox(phone_rgb, src_lin_full.shape[-2], src_lin_full.shape[-1])
    src_small = F.interpolate(src_lin_full.detach(), size=(down, down), mode="area")
    phone_small = F.interpolate(phone_fit.detach(), size=(down, down), mode="area")
    opt = torch.optim.Adam(init_net.parameters(), lr=lr, betas=(0.9, 0.99))
    full_mask = torch.ones((1, 1, down, down), dtype=src_small.dtype, device=src_small.device)
    for it in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pred = init_net(src_small)
        pred_lab = srgb_to_lab_fast(pred)
        phone_lab = srgb_to_lab_fast(phone_small)
        loss_mean = (pred_lab.mean(dim=(2, 3)) - phone_lab.mean(dim=(2, 3))).abs().mean()
        loss_std = (pred_lab.std(dim=(2, 3)) - phone_lab.std(dim=(2, 3))).abs().mean()
        loss_swd = masked_swd_lab(pred_lab, phone_lab, full_mask, max_samples=30000, n_proj=6)
        loss_reg = init_net.du_s.pow(2).mean() + init_net.du_wb.pow(2).mean() + 0.2 * init_net.du_ccm.pow(2).mean() + 0.2 * init_net.du_gamma.pow(2).mean()
        loss = 0.25 * loss_mean + 0.10 * loss_std + 0.35 * loss_swd + 0.01 * loss_reg
        loss.backward()
        opt.step()
        if it % 40 == 0 or it == int(steps) - 1:
            print(f"[init_net {it:03d}] loss={float(loss.detach().item()):.5f} swd={float(loss_swd.detach().item()):.5f}")


def range_penalty(x: torch.Tensor, low=0.0, high=1.0, low_weight=1.0, high_weight=1.0):
    return low_weight * F.relu(float(low) - x).pow(2).mean() + high_weight * F.relu(x - float(high)).pow(2).mean()


def isp_range_loss(dbg: dict):
    total = dbg["final"].new_tensor(0.0)
    specs = [("after_exposure", 0, 8, 0.2, 0.02), ("after_wb", 0, 8, 0.3, 0.03), ("after_ccm", 0, 6, 1.0, 0.15), ("after_tone", 0, 1.2, 1.0, 0.5), ("after_saturation", 0, 1.1, 1.0, 0.5)]
    for name, low, high, lw, hw in specs:
        if name in dbg:
            total = total + range_penalty(dbg[name], low, high, lw, hw)
    return total


def range_stats(x: torch.Tensor):
    x = x.detach()
    return {"min": float(x.min()), "max": float(x.max()), "lt0": float((x < 0).float().mean()), "gt1": float((x > 1).float().mean()), "gt4": float((x > 4).float().mean()), "gt8": float((x > 8).float().mean())}


class ResidualISPV4(nn.Module):
    def __init__(self, pattern="RGGB", ccm_clamp_abs=2.0, lut_size=24, use_lut=True, lut_black_anchor=True, lut_near_black_fade=True, lut_near_black_low=0.030, lut_near_black_high=0.150, raw_denoiser=None, denoiser_exp_min_log2=None, denoiser_exp_max_log2=None, use_raw_denoiser=True, use_denoiser_clean_prior=True, use_denoiser_uncertainty=True, uncertainty_confidence_mode="reliability", uncertainty_temperature=4.0, ltm_noise_aware=True, ltm_reliability_power=1.0, ltm_structure_power=1.0, ltm_highlight_guard=0.5, ltm_min_gate=0.20, gtm_luminance_only=False, ltm_luminance_only=False):
        super().__init__()
        self.pattern = pattern
        self.ccm_clamp_abs = float(ccm_clamp_abs)
        self.raw_denoiser = raw_denoiser
        self.use_raw_denoiser = bool(use_raw_denoiser and raw_denoiser is not None)
        self.use_denoiser_clean_prior = bool(use_denoiser_clean_prior and self.use_raw_denoiser)
        self.use_denoiser_uncertainty = bool(use_denoiser_uncertainty and self.use_raw_denoiser)
        self.denoiser_exp_min_log2 = float(denoiser_exp_min_log2 if denoiser_exp_min_log2 is not None else math.log2(50.0))
        self.denoiser_exp_max_log2 = float(denoiser_exp_max_log2 if denoiser_exp_max_log2 is not None else math.log2(200.0))
        self.uncertainty_confidence_mode = str(uncertainty_confidence_mode).lower()
        self.uncertainty_temperature = float(uncertainty_temperature)
        self.ltm_noise_aware = bool(ltm_noise_aware)
        self.ltm_reliability_power = float(ltm_reliability_power)
        self.ltm_structure_power = float(ltm_structure_power)
        self.ltm_highlight_guard = float(ltm_highlight_guard)
        self.ltm_min_gate = float(ltm_min_gate)
        self.gtm_luminance_only = bool(gtm_luminance_only)
        self.ltm_luminance_only = bool(ltm_luminance_only)
        self.exposure_ev_range = 2.0
        self.register_buffer("s0", torch.tensor(1.0))
        self.register_buffer("log2_s0", torch.tensor(0.0))
        self.register_buffer("wb0_rb", torch.tensor([1.0, 1.0]))
        self.register_buffer("ccm0", torch.eye(3))
        self.register_buffer("gtm0", torch.tensor([1.15, 1.20, 1.00]))
        self.register_buffer("sat0", torch.tensor(1.2))
        self.du_s = nn.Parameter(torch.zeros(1))
        self.du_wb = nn.Parameter(torch.zeros(2))
        self.du_ccm = nn.Parameter(torch.zeros(3, 3))
        self.du_gtm = nn.Parameter(torch.zeros(3))
        self.du_sat = nn.Parameter(torch.zeros(1))
        self.ltm_maps = nn.Parameter(torch.zeros(1, 5, 16, 16))
        self.lut = LUT3D(size=lut_size) if use_lut else None
        if self.lut is not None and isinstance(self.lut, LUT3D):
            self.lut.black_anchor = bool(lut_black_anchor)
            self.lut.near_black_fade = bool(lut_near_black_fade)
            self.lut.near_black_low = float(lut_near_black_low)
            self.lut.near_black_high = float(lut_near_black_high)
        self._raw_frontend_cache = None
        self._raw_frontend_cache_info = {}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        lut_key = prefix + "lut.lut"
        if self.lut is not None and lut_key in state_dict:
            loaded_lut = state_dict[lut_key]
            if (
                loaded_lut.ndim == 4
                and loaded_lut.shape[-1] == 3
                and (
                    not isinstance(self.lut, LUT3D)
                    or tuple(self.lut.lut.shape) != tuple(loaded_lut.shape)
                )
            ):
                self.lut = LUT3D(size=int(loaded_lut.shape[0]))
            elif (
                loaded_lut.ndim == 4
                and loaded_lut.shape[-1] == 2
                and (
                    not isinstance(self.lut, YCbCrChromaLUT3D)
                    or tuple(self.lut.lut.shape) != tuple(loaded_lut.shape)
                )
            ):
                self.lut = YCbCrChromaLUT3D(size=int(loaded_lut.shape[0]))
            elif (
                loaded_lut.ndim == 3
                and loaded_lut.shape[-1] == 2
                and (
                    not isinstance(self.lut, YCbCrChromaLUT2D)
                    or tuple(self.lut.lut.shape) != tuple(loaded_lut.shape)
                )
            ):
                self.lut = YCbCrChromaLUT2D(size=int(loaded_lut.shape[0]))
        log_key = prefix + "log2_s0"
        s0_key = prefix + "s0"
        du_key = prefix + "du_s"
        if log_key not in state_dict and s0_key in state_dict:
            s0 = state_dict[s0_key].detach().float()
            du = state_dict.get(du_key, torch.zeros_like(s0)).detach().float()
            legacy_gain = (s0 + torch.tanh(du) * 0.5).clamp(0.1, 35.0)
            state_dict[log_key] = torch.log2(legacy_gain).to(state_dict[s0_key])
            state_dict[s0_key] = legacy_gain.to(state_dict[s0_key])
            if du_key in state_dict:
                state_dict[du_key] = torch.zeros_like(state_dict[du_key])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def set_base_exposure(self, gain):
        gain_t = torch.as_tensor(gain, device=self.log2_s0.device, dtype=self.log2_s0.dtype).detach().clamp(0.1, 35.0)
        gain_t = gain_t.reshape_as(self.log2_s0)
        self.log2_s0.copy_(torch.log2(gain_t))
        self.s0.copy_(gain_t.to(self.s0))

    @torch.no_grad()
    def set_effective_exposure(self, gain):
        self.set_base_exposure(gain)
        self.du_s.zero_()

    def exposure(self):
        ev = self.log2_s0.to(self.du_s) + float(self.exposure_ev_range) * torch.tanh(self.du_s)
        gain = torch.pow(self.du_s.new_tensor(2.0), ev)
        return gain.clamp(0.1, 35.0).view(1, 1, 1, 1)

    def wb_gain(self):
        wb = (self.wb0_rb + torch.tanh(self.du_wb) * 0.8).clamp(0.2, 4.0)
        return torch.stack([wb[0], torch.ones_like(wb[0]), wb[1]]).view(1, 3, 1, 1)

    def ccm(self):
        M = (self.ccm0 + torch.tanh(self.du_ccm) * 0.25).clamp(-self.ccm_clamp_abs, self.ccm_clamp_abs)
        return M / M.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def saturation(self):
        return (self.sat0 + torch.tanh(self.du_sat) * 0.6).clamp(0.0, 3.0).view(1, 1, 1, 1)

    def gtm_params(self, dtype, device):
        d = torch.tanh(self.du_gtm)
        base = self.gtm0.to(dtype=dtype, device=device)
        return (base[0] + d[0] * 0.6).clamp(0.4, 2.5), (base[1] + d[1] * 0.6).clamp(0.4, 2.5), (base[2] + d[2] * 0.8).clamp(0.2, 2.5)

    def f_tm(self, x, a, b, c):
        eps = 1e-8
        x = x.clamp(0, 1)
        xa = (x + eps).pow(a)
        yb = (1.0 - x + eps).pow(b)
        return (xa / (xa + c * yb + eps)).clamp(0, 1)

    def apply_luminance_curve(self, rgb, a, b, c):
        w = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        y = (rgb * w).sum(1, keepdim=True).clamp_min(1e-4)
        y_tm = self.f_tm(y, a, b, c)
        scale = (y_tm / y).clamp(0.25, 4.0)
        return (rgb * scale).clamp(0, 1)

    def apply_gtm(self, rgb_gain):
        a, b, c = self.gtm_params(rgb_gain.dtype, rgb_gain.device)
        if self.gtm_luminance_only:
            return self.apply_luminance_curve(rgb_gain, a, b, c)
        return self.f_tm(rgb_gain, a, b, c)

    def make_guidance(self, rgb_gain, rgb_gtm):
        w = rgb_gain.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        y_gain = (rgb_gain * w).sum(1, keepdim=True).clamp(1e-6, 1.0)
        y_gtm = (rgb_gtm * w).sum(1, keepdim=True).clamp(1e-6, 1.0)
        detail = (y_gain - gaussian_blur(y_gain, kernel_size=31, sigma=7.0)).abs()
        g = 0.5 * (y_gain - y_gtm).abs() + 0.5 * detail
        return (g / (g.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)).clamp(0, 1)

    def apply_ltm(self, rgb_gain, rgb_gtm, raw_reliability=None, return_debug=False):
        _, _, H, W = rgb_gain.shape
        guide = self.make_guidance(rgb_gain, rgb_gtm)
        maps = torch.tanh(F.interpolate(self.ltm_maps, size=(H, W), mode="bilinear", align_corners=False))
        a0, b0, c0 = self.gtm_params(rgb_gain.dtype, rgb_gain.device)
        A = (a0 * (1.0 + 0.25 * maps[:, 0:1] * (2.0 * guide - 1.0))).clamp(0.4, 2.8)
        Bp = (b0 * (1.0 + 0.25 * maps[:, 1:2] * (2.0 * guide - 1.0))).clamp(0.4, 2.8)
        C = (c0 * (1.0 + 0.20 * maps[:, 2:3] * (2.0 * guide - 1.0))).clamp(0.2, 2.8)
        G_ltm = (1.0 + 0.15 * maps[:, 3:4] * (2.0 * guide - 1.0)).clamp(0.7, 1.3)
        rgb_local = self.f_tm((rgb_gain * G_ltm).clamp(0, 1), A, Bp, C)
        W_ltm = torch.sigmoid(2.0 * maps[:, 4:5] + 2.0 * (guide - 0.5))
        min_gate = float(np.clip(self.ltm_min_gate, 0.0, 1.0))
        structure_gate = min_gate + (1.0 - min_gate) * guide.pow(max(self.ltm_structure_power, 1e-6))
        if raw_reliability is None:
            noise_gate = torch.ones_like(W_ltm)
        else:
            noise_gate = raw_reliability.detach().clamp(0, 1)
            if noise_gate.shape[-2:] != (H, W):
                noise_gate = F.interpolate(noise_gate, size=(H, W), mode="bilinear", align_corners=False)
            noise_gate = min_gate + (1.0 - min_gate) * noise_gate.pow(max(self.ltm_reliability_power, 1e-6))
        wp = rgb_gain.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        y_gain = (rgb_gain.detach().clamp(0, 1) * wp).sum(1, keepdim=True)
        highlight_gate = (1.0 - self.ltm_highlight_guard) + self.ltm_highlight_guard * (1.0 - y_gain).clamp(0, 1).sqrt()
        W_final = (W_ltm * structure_gate * noise_gate * highlight_gate).clamp(0, 1) if self.ltm_noise_aware else W_ltm
        if self.ltm_luminance_only:
            y_global = (rgb_gtm.clamp(0, 1) * wp).sum(1, keepdim=True).clamp_min(1e-4)
            y_local = (rgb_local.clamp(0, 1) * wp).sum(1, keepdim=True)
            y_out = ((1.0 - W_final) * y_global + W_final * y_local).clamp(0, 1)
            rgb_out = (rgb_gtm * (y_out / y_global).clamp(0.25, 4.0)).clamp(0, 1)
        else:
            rgb_out = ((1.0 - W_final) * rgb_gtm + W_final * rgb_local).clamp(0, 1)
        extras = {"ltm_final_weight": W_final, "ltm_structure_gate": structure_gate, "ltm_noise_gate": noise_gate, "ltm_highlight_gate": highlight_gate}
        if return_debug:
            return rgb_out, guide, W_final, G_ltm, extras
        return rgb_out, guide, W_final, G_ltm

    def apply_ccm(self, rgb):
        B, C, H, W = rgb.shape
        y = rgb.permute(0, 2, 3, 1).reshape(-1, 3) @ self.ccm().to(rgb).T
        return y.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def apply_saturation(self, rgb):
        w = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        y = (rgb * w).sum(1, keepdim=True)
        highlight = ((y.clamp(0.0, 1.0) - 0.75) / 0.20).clamp(0.0, 1.0)
        highlight = highlight * highlight * (3.0 - 2.0 * highlight)
        sat = 1.0 + (self.saturation() - 1.0) * (1.0 - highlight)
        return y + sat * (rgb - y)

    def clear_raw_frontend_cache(self):
        self._raw_frontend_cache = None
        self._raw_frontend_cache_info = {}

    def _compute_raw_denoiser(self, raw_packed, exp_t):
        if not self.use_raw_denoiser or self.raw_denoiser is None:
            zero = torch.zeros_like(raw_packed); one = torch.ones_like(raw_packed)
            return {"clean": raw_packed, "log_var": zero, "variance": zero, "sigma": zero, "phys_var": zero, "relative_noise": zero, "reliability": one, "local_signal": raw_packed.clamp_min(1 / 4095.0), "residual": zero}
        cond = make_condition_tensor(exp_t, self.denoiser_exp_min_log2, self.denoiser_exp_max_log2, device=raw_packed.device, dtype=raw_packed.dtype)
        den = complete_denoiser_output(self.raw_denoiser(raw_packed, cond))
        if "relative_noise" not in den or "reliability" not in den:
            local_signal = F.avg_pool2d(F.pad(den["clean"].detach().clamp(0, 1), (4, 4, 4, 4), mode="reflect"), 9, stride=1).clamp_min(1 / 4095.0)
            relative_noise = (den["sigma"] / local_signal).clamp(0, 100)
            med = relative_noise.detach().flatten(1).median(dim=1).values.view(-1, 1, 1, 1).clamp_min(1e-6)
            den["relative_noise"] = relative_noise
            den["reliability"] = (1.0 / (1.0 + relative_noise / (4.0 * med))).clamp(0, 1)
            den["local_signal"] = local_signal
        return den

    def _run_raw_denoiser(self, raw_packed, exp_t):
        if self._raw_frontend_cache is not None:
            return dict(self._raw_frontend_cache["den"])
        return self._compute_raw_denoiser(raw_packed, exp_t)

    def _raw_confidence(self, den):
        rel = den.get("relative_noise", None)
        if rel is None or float(rel.detach().max()) <= 0:
            return unpack_bayer_tensor(den["reliability"].clamp(0, 1), pattern=self.pattern)
        rel_raw = unpack_bayer_tensor(rel.clamp_min(0), pattern=self.pattern)
        med = rel_raw.flatten(1).median(dim=1).values.view(-1, 1, 1, 1).clamp_min(1e-6)
        return (1.0 / (1.0 + rel_raw / (med * max(self.uncertainty_temperature, 1e-6)))).clamp(0, 1)

    @torch.no_grad()
    def cache_raw_frontend(self, raw, exp_t):
        self.clear_raw_frontend_cache()
        raw_packed = pack_bayer_tensor(raw, pattern=self.pattern)
        den = self._compute_raw_denoiser(raw_packed, exp_t)
        raw_dn_packed = (den["clean"] if self.use_denoiser_clean_prior else raw_packed).clamp(0, 1)
        raw_denoised_full = unpack_bayer_tensor(raw_dn_packed, pattern=self.pattern).clamp(0, 1).detach()
        raw_variance_full = unpack_bayer_tensor(den["variance"].clamp_min(0), pattern=self.pattern).detach()
        raw_confidence = (
            self._raw_confidence(den)
            if self.use_denoiser_uncertainty
            else unpack_bayer_tensor(torch.ones_like(raw_packed), pattern=self.pattern)
        ).detach()
        linear_rgb = demosaic_malvar2004(raw_denoised_full, pattern=self.pattern).clamp(0, 1).detach()
        den_cached = {k: v.detach() if torch.is_tensor(v) else v for k, v in den.items()}
        self._raw_frontend_cache = {
            "den": den_cached,
            "raw_denoised_full": raw_denoised_full,
            "raw_variance_full": raw_variance_full,
            "raw_confidence": raw_confidence,
            "linear_rgb": linear_rgb,
        }
        tensors = [v for v in self._raw_frontend_cache.values() if torch.is_tensor(v)]
        tensors.extend(v for v in den_cached.values() if torch.is_tensor(v))
        bytes_total = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
        exp_value = float(exp_t.detach().cpu().reshape(-1)[0]) if torch.is_tensor(exp_t) else float(exp_t)
        self._raw_frontend_cache_info = {
            "enabled": True,
            "raw_shape": list(raw.shape),
            "packed_shape": list(raw_packed.shape),
            "exposure_t_us": exp_value,
            "memory_mb": bytes_total / (1024.0 * 1024.0),
            "linear_rgb_mean": float(linear_rgb.detach().mean().cpu()),
        }
        return dict(self._raw_frontend_cache_info)

    def _cached_frontend(self):
        return self._raw_frontend_cache

    @torch.no_grad()
    def denoise_raw(self, raw, exp_t):
        if not self.use_denoiser_clean_prior:
            return raw.clamp(0, 1)
        cache = self._cached_frontend()
        if cache is not None:
            return cache["raw_denoised_full"]
        den = self._run_raw_denoiser(pack_bayer_tensor(raw, pattern=self.pattern), exp_t)
        return unpack_bayer_tensor(den["clean"], pattern=self.pattern).clamp(0, 1)

    def render(self, raw, exp_t):
        cache = self._cached_frontend()
        if cache is not None:
            raw_conf = cache["raw_confidence"]
            rgb = cache["linear_rgb"]
        else:
            raw_packed = pack_bayer_tensor(raw, pattern=self.pattern)
            den = self._run_raw_denoiser(raw_packed, exp_t)
            raw_conf = self._raw_confidence(den) if self.use_denoiser_uncertainty else unpack_bayer_tensor(torch.ones_like(raw_packed), pattern=self.pattern)
            raw_dn_packed = den["clean"] if self.use_denoiser_clean_prior else raw_packed
            raw_dn = unpack_bayer_tensor(raw_dn_packed, pattern=self.pattern).clamp(0, 1)
            rgb = demosaic_malvar2004(raw_dn, pattern=self.pattern)
        rgb = rgb * self.exposure() * self.wb_gain()
        rgb = self.apply_ccm(rgb).clamp_min(0)
        rgb_gain = rgb
        rgb_gtm = self.apply_gtm(rgb_gain)
        rgb, _, _, _ = self.apply_ltm(rgb_gain, rgb_gtm, raw_reliability=raw_conf)
        if self.lut is not None:
            rgb = self.lut(rgb).clamp(0, 1)
        rgb = self.apply_saturation(rgb).clamp(0, 1)
        return rgb

    def forward(self, raw, exp_t):
        return self.render(raw, exp_t).clamp(0, 1)

    def forward_debug(self, raw, exp_t):
        out = {}
        cache = self._cached_frontend()
        if cache is not None:
            out["raw_denoised_full"] = cache["raw_denoised_full"]
            out["raw_variance_full"] = cache["raw_variance_full"]
            out["raw_confidence"] = cache["raw_confidence"]
            raw_conf = out["raw_confidence"]
            rgb_linear = cache["linear_rgb"]
        else:
            raw_packed = pack_bayer_tensor(raw, pattern=self.pattern)
            den = self._run_raw_denoiser(raw_packed, exp_t)
            raw_dn = (den["clean"] if self.use_denoiser_clean_prior else raw_packed).clamp(0, 1)
            raw_conf = self._raw_confidence(den) if self.use_denoiser_uncertainty else unpack_bayer_tensor(torch.ones_like(raw_packed), pattern=self.pattern)
            out["raw_denoised_full"] = unpack_bayer_tensor(raw_dn, pattern=self.pattern)
            out["raw_variance_full"] = unpack_bayer_tensor(den["variance"].clamp_min(0), pattern=self.pattern)
            out["raw_confidence"] = raw_conf
            rgb_linear = demosaic_malvar2004(out["raw_denoised_full"], pattern=self.pattern)
        out["linear_rgb"] = rgb_linear
        out["after_exposure"] = rgb_linear * self.exposure()
        out["after_wb"] = out["after_exposure"] * self.wb_gain()
        out["after_ccm"] = self.apply_ccm(out["after_wb"]).clamp_min(0)
        rgb_gtm = self.apply_gtm(out["after_ccm"])
        out["after_gtm"] = rgb_gtm
        rgb_ltm, guide, w_ltm, g_ltm, ltm_extra = self.apply_ltm(out["after_ccm"], rgb_gtm, raw_reliability=raw_conf, return_debug=True)
        out["after_ltm"] = rgb_ltm
        out["after_tone"] = rgb_ltm
        out.update(ltm_extra)
        out["before_lut"] = rgb_ltm.clamp(0, 1)
        out["after_lut"] = self.lut(out["before_lut"]).clamp(0, 1) if self.lut is not None else out["before_lut"]
        out["after_saturation"] = self.apply_saturation(out["after_lut"]).clamp(0, 1)
        out["final"] = out["after_saturation"]
        out["range_stats"] = {k: range_stats(out[k]) for k in ("after_wb", "after_ccm", "after_tone", "after_saturation", "before_lut", "after_lut") if k in out}
        return out

    def ltm_regularization(self, smooth_weight=1e-3, weight_bias_weight=1e-4):
        x = self.ltm_maps
        return smooth_weight * (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + smooth_weight * (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + weight_bias_weight * x[:, 4:5].abs().mean()


def init_exposure_wb_base(
    isp,
    src_lin,
    tgt_srgb,
    valid_mask,
    wb_mode: str = "ratio",
    wb_stat: str = "median",
    wb_min: float = 0.3,
    wb_max: float = 3.0,
):
    with torch.no_grad():
        tgt_lin = srgb_to_linear(tgt_srgb).clamp(0, 1)
        w = src_lin.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        Ys, Yt = (src_lin * w).sum(1, keepdim=True), (tgt_lin * w).sum(1, keepdim=True)
        sel = (valid_mask > 0.2) & (Yt > 0.05) & (Yt < 0.8) & torch.isfinite(Yt) & torch.isfinite(Ys)
        if sel.sum() < 1024:
            sel = valid_mask > 0.05
        s0 = (Yt[sel].mean() / (Ys[sel].mean() + 1e-6)).clamp(0.25, 35.0)
        isp.set_base_exposure(s0)
        Rs, Gs, Bs = src_lin[:, 0:1], src_lin[:, 1:2], src_lin[:, 2:3]
        Rt, Gt, Bt = tgt_lin[:, 0:1], tgt_lin[:, 1:2], tgt_lin[:, 2:3]
        phone_rg = _masked_channel_ratio(Rt, Gt, sel, wb_stat)
        phone_bg = _masked_channel_ratio(Bt, Gt, sel, wb_stat)
        if str(wb_mode).lower() == "phone":
            phone_sel = _phone_neutral_selector(tgt_srgb, sel, min_count=1024)
            wr0 = _masked_channel_ratio(tgt_srgb[:, 0:1], tgt_srgb[:, 1:2], phone_sel, wb_stat).clamp(float(wb_min), float(wb_max))
            wb0 = _masked_channel_ratio(tgt_srgb[:, 2:3], tgt_srgb[:, 1:2], phone_sel, wb_stat).clamp(float(wb_min), float(wb_max))
        elif str(wb_mode).lower() == "ratio":
            wr0 = (phone_rg / (_masked_channel_ratio(Rs, Gs, sel, wb_stat) + 1e-6)).clamp(float(wb_min), float(wb_max))
            wb0 = (phone_bg / (_masked_channel_ratio(Bs, Gs, sel, wb_stat) + 1e-6)).clamp(float(wb_min), float(wb_max))
        else:
            raise ValueError(f"Unsupported init_wb_mode: {wb_mode}")
        isp.wb0_rb.copy_(torch.stack([wr0, wb0]).to(isp.wb0_rb))


def neutral_ratio_loss(pred_srgb: torch.Tensor, tgt_srgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    tgt_lab = srgb_to_lab_fast(tgt_srgb.clamp(0, 1))
    chroma = torch.sqrt(tgt_lab[:, 1:2].pow(2) + tgt_lab[:, 2:3].pow(2))
    tgt_y = _rgb_luminance(srgb_to_linear(tgt_srgb.clamp(0, 1)))
    neutral = ((chroma < 10.0) & (tgt_y > 0.05) & (tgt_y < 0.85)).float() * mask.detach().clamp(0, 1)
    if float(neutral.detach().sum()) < 64:
        return pred_srgb.new_tensor(0.0)

    eps = 1.0 / 255.0
    pred_lin = srgb_to_linear(pred_srgb.clamp(0, 1)).clamp_min(eps)
    tgt_lin = srgb_to_linear(tgt_srgb.clamp(0, 1)).clamp_min(eps)
    pred_rg = torch.log(pred_lin[:, 0:1] / pred_lin[:, 1:2])
    pred_bg = torch.log(pred_lin[:, 2:3] / pred_lin[:, 1:2])
    tgt_rg = torch.log(tgt_lin[:, 0:1] / tgt_lin[:, 1:2])
    tgt_bg = torch.log(tgt_lin[:, 2:3] / tgt_lin[:, 1:2])
    loss = (pred_rg - tgt_rg).abs() + (pred_bg - tgt_bg).abs()
    return (loss * neutral).sum() / (neutral.sum() * 2.0 + 1e-6)
