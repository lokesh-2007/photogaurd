"""
PhotoGuard Engine — Lightweight MIT PhotoGuard Encoder Attack (Low-VRAM FP16)
=============================================================================
Refactored from MIT MadryLab's official PhotoGuard encoder/latent-space attack
(https://github.com/MadryLab/photoguard) for production web backends.

Key Design Goals:
- Zero Notebook/Plotting Bloat: Pure PyTorch + Diffusers Tiny AutoEncoder (`TAESD`).
- Ultra-Low VRAM (< 500 MB): Loads `madebyollin/taesd` (and `taesdxl` cross-model
  target) in `torch.float16` on CUDA/MPS with frozen weights, eliminating the
  multi-gigabyte UNet / CLIP / full SD-VAE graph unless explicitly opted in.
- Strict L-Infinity Budget: Hard projection clamp (`epsilon` typically 4/255–6/255)
  with optional perceptual JND protection for uniform paper and sharp text edges.
- EOT (Expectation Over Transformation): Random differentiable scaling (95%-105%),
  Gaussian blur, color jitter, and Straight-Through Estimator (STE) 8x8 DCT JPEG
  simulation inside each optimization step.
- Lossless 32-Bit RGBA PNG Output: Byte-stream wrapper strips EXIF/GPS metadata and
  returns an uncompressed 32-bit RGBA PNG `io.BytesIO` buffer (`compress_level=0`).

FastAPI `/api/protect` Integration Example:
-------------------------------------------
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from photoguard_engine import PhotoGuardEngine

engine: PhotoGuardEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = PhotoGuardEngine.from_pretrained()
    yield
    engine = None

app = FastAPI(lifespan=lifespan)

@app.post("/api/protect")
async def protect_image(
    file: UploadFile = File(...),
    epsilon: float = Form(6.0 / 255.0),
    steps: int = Form(30),
    enable_eot: bool = Form(True),
):
    raw_bytes = await file.read()
    png_buf, metrics = engine.protect_image_bytes(
        raw_bytes, epsilon=epsilon, steps=steps, enable_eot=enable_eot
    )
    return StreamingResponse(
        png_buf,
        media_type="image/png",
        headers={
            "X-PSNR": str(metrics["psnr"]),
            "X-SSIM": str(metrics["ssim"]),
            "X-Process-Time-Ms": str(metrics["elapsed_ms"]),
        },
    )
```
"""

from __future__ import annotations

import io
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderTiny
from PIL import Image, ImageOps

DEFAULT_MAX_LONG_EDGE = 1024
DEFAULT_EPSILON = 6.0 / 255.0
DEFAULT_STEPS = 30

# Standard JPEG Luminance & Chrominance 8x8 Quantization Tables
_JPEG_LUMA_Q = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

_JPEG_CHROMA_Q = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def _build_dct_matrix() -> torch.Tensor:
    mat = torch.zeros((8, 8), dtype=torch.float32)
    for k in range(8):
        alpha = math.sqrt(1.0 / 8.0) if k == 0 else math.sqrt(2.0 / 8.0)
        for n in range(8):
            mat[k, n] = alpha * math.cos(math.pi * (2.0 * n + 1.0) * k / 16.0)
    return mat


def _freeze_eval(module: nn.Module, device: str, dtype: torch.dtype) -> nn.Module:
    module = module.to(device=device, dtype=dtype).eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator rounding for differentiable JPEG quantization."""
    return x + (x.round() - x).detach()


def _feature_ssim(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """Differentiable per-channel SSIM computed in latent/feature space."""
    a = feat_a.float()
    b = feat_b.float()
    mu_a = a.mean(dim=(-2, -1), keepdim=True)
    mu_b = b.mean(dim=(-2, -1), keepdim=True)
    var_a = ((a - mu_a) ** 2).mean(dim=(-2, -1), keepdim=True)
    var_b = ((b - mu_b) ** 2).mean(dim=(-2, -1), keepdim=True)
    cov = ((a - mu_a) * (b - mu_b)).mean(dim=(-2, -1), keepdim=True)
    c1, c2 = 1e-4, 9e-4
    ssim_map = ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    )
    return ssim_map.mean()


def compute_jnd_bound(x_neg1_1: torch.Tensor, eps_tensor: float) -> torch.Tensor:
    """
    Perceptual Just-Noticeable-Difference (JND) L-infinity envelope capped by `eps_tensor`.
    Attenuates perturbation on uniform white paper and crisp text borders so documents
    remain visually spotless while textured areas receive the full epsilon budget.
    """
    x_01 = (x_neg1_1.float() + 1.0) * 0.5
    luma = 0.299 * x_01[:, 0:1] + 0.587 * x_01[:, 1:2] + 0.114 * x_01[:, 2:3]
    local_max = F.max_pool2d(luma, kernel_size=3, stride=1, padding=1)
    local_min = -F.max_pool2d(-luma, kernel_size=3, stride=1, padding=1)
    local_range = local_max - local_min

    is_uniform_paper = (local_range < 0.035) & ((luma > 0.88) | (luma < 0.08))
    is_sharp_text_edge = local_range > 0.28

    paper_bound = torch.full_like(luma, min(eps_tensor, (2.0 / 255.0) * 2.0))
    text_bound = torch.full_like(luma, min(eps_tensor * 0.45, (3.2 / 255.0) * 2.0))
    texture_bound = torch.clamp(
        paper_bound + (eps_tensor - paper_bound) * (local_range * 3.0),
        max=eps_tensor,
    )

    bound = torch.where(
        is_uniform_paper,
        paper_bound,
        torch.where(is_sharp_text_edge, text_bound, texture_bound),
    )
    return bound.repeat(1, 3, 1, 1).to(dtype=x_neg1_1.dtype)


def compute_psnr_ssim(orig: Image.Image, prot: Image.Image) -> tuple[float, float]:
    """Computes exact RGB PSNR (dB) and luminance SSIM between two PIL Images."""
    a = np.asarray(orig.convert("RGB"), dtype=np.float64)
    b = np.asarray(prot.convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    psnr = 99.9 if mse < 1e-10 else float(10.0 * np.log10((255.0 ** 2) / mse))

    la = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = la.mean(), lb.mean()
    var_a, var_b = la.var(), lb.var()
    cov = np.mean((la - mu_a) * (lb - mu_b))
    ssim = float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    )
    return round(psnr, 2), round(max(0.0, min(1.0, ssim)), 4)


def encode_lossless_png_stream(img: Image.Image) -> io.BytesIO:
    """
    Converts a PIL Image to raw 32-bit RGBA (stripping all EXIF/GPS/IPTC tags)
    and serializes it as an uncompressed lossless PNG byte stream.
    """
    rgba = Image.fromarray(np.asarray(img.convert("RGB"), dtype=np.uint8), mode="RGB").convert("RGBA")
    out_buf = io.BytesIO()
    rgba.save(out_buf, format="PNG", compress_level=0)
    out_buf.seek(0)
    return out_buf


class PhotoGuardEngine:
    """
    Streamlined, low-VRAM (<500 MB) MIT PhotoGuard adversarial encoder attack engine.
    Targets `madebyollin/taesd` (and optionally `madebyollin/taesdxl`) in FP16.
    """

    def __init__(
        self,
        taesd: AutoencoderTiny,
        taesdxl: AutoencoderTiny | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.taesd = taesd
        self.taesdxl = taesdxl
        self.device = device
        self.dtype = dtype
        self.dct_mat = _build_dct_matrix().to(device=device, dtype=torch.float32)
        self.luma_q = _JPEG_LUMA_Q.to(device=device, dtype=torch.float32)
        self.chroma_q = _JPEG_CHROMA_Q.to(device=device, dtype=torch.float32)

    @classmethod
    def from_pretrained(
        cls,
        primary_id: str = "madebyollin/taesd",
        secondary_id: str | None = "madebyollin/taesdxl",
        device: str | None = None,
    ) -> "PhotoGuardEngine":
        resolved_device = device or (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        dtype = torch.float16 if resolved_device in ("cuda", "mps") else torch.float32

        taesd = _freeze_eval(
            AutoencoderTiny.from_pretrained(primary_id, torch_dtype=dtype),
            device=resolved_device,
            dtype=dtype,
        )

        taesdxl = None
        if secondary_id:
            try:
                taesdxl = _freeze_eval(
                    AutoencoderTiny.from_pretrained(secondary_id, torch_dtype=dtype),
                    device=resolved_device,
                    dtype=dtype,
                )
            except Exception:
                taesdxl = None

        return cls(taesd=taesd, taesdxl=taesdxl, device=resolved_device, dtype=dtype)

    def _differentiable_jpeg_sim(self, x_neg1_1: torch.Tensor, quality: int) -> torch.Tensor:
        orig_dtype = x_neg1_1.dtype
        x = ((x_neg1_1.float() + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)
        b, _, h, w = x.shape

        r, g, bl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * bl
        cb = -0.168736 * r - 0.331264 * g + 0.5 * bl + 128.0
        cr = 0.5 * r - 0.418688 * g - 0.081312 * bl + 128.0

        cb_sub = F.interpolate(
            F.avg_pool2d(cb, kernel_size=2, stride=2), size=(h, w), mode="bilinear", align_corners=False
        )
        cr_sub = F.interpolate(
            F.avg_pool2d(cr, kernel_size=2, stride=2), size=(h, w), mode="bilinear", align_corners=False
        )

        q = max(10, min(95, quality))
        q_scale = (5000.0 / q) / 100.0 if q < 50 else (200.0 - 2.0 * q) / 100.0
        luma_q = torch.clamp((self.luma_q * q_scale).round(), min=1.0)
        chroma_q = torch.clamp((self.chroma_q * q_scale).round(), min=1.0)

        def quantize_channel(ch: torch.Tensor, q_table: torch.Tensor) -> torch.Tensor:
            pad_h = (8 - (h % 8)) % 8
            pad_w = (8 - (w % 8)) % 8
            if pad_h or pad_w:
                ch = F.pad(ch, (0, pad_w, 0, pad_h), mode="replicate")
            _, _, ph, pw = ch.shape
            blocks = ch.unfold(2, 8, 8).unfold(3, 8, 8)
            coeff = self.dct_mat @ (blocks - 128.0) @ self.dct_mat.t()
            quant = _round_ste(coeff / q_table) * q_table
            recon = (self.dct_mat.t() @ quant @ self.dct_mat) + 128.0
            recon = recon.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, 1, ph, pw)
            return recon[:, :, :h, :w]

        yq = quantize_channel(y, luma_q)
        cbq = quantize_channel(cb_sub, chroma_q)
        crq = quantize_channel(cr_sub, chroma_q)

        cb_shift = cbq - 128.0
        cr_shift = crq - 128.0
        rgb = torch.cat(
            [
                yq + 1.402 * cr_shift,
                yq - 0.344136 * cb_shift - 0.714136 * cr_shift,
                yq + 1.772 * cb_shift,
            ],
            dim=1,
        ).clamp(0.0, 255.0)
        return ((rgb / 255.0) * 2.0 - 1.0).to(dtype=orig_dtype)

    def apply_eot_transform(self, x_neg1_1: torch.Tensor) -> torch.Tensor:
        """
        Differentiable EOT step: random 95%-105% scaling, slight Gaussian blur,
        mild color jitter, and simulated 8x8 DCT JPEG compression.
        """
        _, _, h, w = x_neg1_1.shape

        scale = float(np.random.uniform(0.95, 1.05))
        nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
        scaled = F.interpolate(x_neg1_1, size=(nh, nw), mode="bilinear", align_corners=False)
        if nh >= h and nw >= w:
            top, left = (nh - h) // 2, (nw - w) // 2
            x_t = scaled[:, :, top : top + h, left : left + w]
        else:
            pt = (h - nh) // 2
            pl = (w - nw) // 2
            x_t = F.pad(scaled, (pl, w - nw - pl, pt, h - nh - pt), mode="reflect")

        if np.random.rand() < 0.65:
            sigma = float(np.random.uniform(0.3, 0.75))
            coords = torch.tensor([-1.0, 0.0, 1.0], device=x_neg1_1.device, dtype=x_neg1_1.dtype)
            g1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
            g1d = g1d / g1d.sum()
            kernel2d = (g1d[:, None] * g1d[None, :]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            x_t = F.conv2d(x_t, kernel2d, padding=1, groups=3)

        if np.random.rand() < 0.7:
            gain = torch.tensor(
                np.random.uniform(0.96, 1.04, size=(1, 3, 1, 1)),
                device=x_neg1_1.device,
                dtype=x_neg1_1.dtype,
            )
            bias = torch.tensor(
                np.random.uniform(-0.03, 0.03, size=(1, 3, 1, 1)),
                device=x_neg1_1.device,
                dtype=x_neg1_1.dtype,
            )
            x_t = (x_t * gain + bias).clamp(-1.0, 1.0)

        if np.random.rand() < 0.75:
            x_t = self._differentiable_jpeg_sim(x_t, quality=int(np.random.randint(78, 94)))

        return x_t

    @staticmethod
    def _extract_encoder_features(
        vae: AutoencoderTiny, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        early_feats: list[torch.Tensor] = []
        handles = []
        layers = getattr(getattr(vae, "encoder", None), "layers", None)
        if layers is not None:
            for idx in (1, 3, 5):
                if idx < len(layers):
                    def _hook(_mod, _inp, out, bucket=early_feats):
                        if isinstance(out, torch.Tensor):
                            bucket.append(out)
                    handles.append(layers[idx].register_forward_hook(_hook))
        try:
            latents = vae.encode(x).latents
        finally:
            for handle in handles:
                handle.remove()
        return latents, early_feats

    def protect_image_tensor(
        self,
        image_tensor: torch.Tensor,
        epsilon: float = DEFAULT_EPSILON,
        steps: int = DEFAULT_STEPS,
        enable_eot: bool = True,
        use_jnd_mask: bool = True,
        input_range: tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """
        Core MIT PhotoGuard-derived encoder attack refactored for low-VRAM FP16 execution.

        Args:
            image_tensor: Input tensor of shape [B, 3, H, W] (typically B=1).
            epsilon: Strict L-infinity pixel perturbation budget in [0, 1] scale
                     (e.g. 4/255 to 6/255).
            steps: Number of Adam-PGD optimization steps.
            enable_eot: Whether to apply EOT transforms inside each PGD iteration.
            use_jnd_mask: Attenuates perturbations on uniform paper & crisp text borders.
            input_range: Expected input value range, either (-1.0, 1.0) or (0.0, 1.0).

        Returns:
            Immunized tensor matching `image_tensor`'s device, dtype, and value range.
        """
        orig_device = image_tensor.device
        orig_dtype = image_tensor.dtype

        if input_range == (0.0, 1.0):
            x_orig = (image_tensor.to(device=self.device, dtype=self.dtype) * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            x_orig = image_tensor.to(device=self.device, dtype=self.dtype).clamp(-1.0, 1.0)

        # Map [0, 1] epsilon to [-1, 1] tensor scale and enforce strict hard L-inf bound
        eps_tensor = float(epsilon) * 2.0
        hard_bound = compute_jnd_bound(x_orig, eps_tensor) if use_jnd_mask else torch.full_like(x_orig, eps_tensor)

        lr_max = eps_tensor * 0.22
        lr_min = eps_tensor * 0.04

        delta = torch.empty_like(x_orig).uniform_(-0.25, 0.25) * hard_bound
        delta.requires_grad_(True)

        with torch.no_grad():
            orig_lat_sd, orig_feats_sd = self._extract_encoder_features(self.taesd, x_orig)
            orig_lat_xl, orig_feats_xl = (
                self._extract_encoder_features(self.taesdxl, x_orig)
                if self.taesdxl is not None
                else (None, [])
            )

        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        m_buf = torch.zeros_like(delta, dtype=torch.float32)
        v_buf = torch.zeros_like(delta, dtype=torch.float32)

        for step in range(1, max(1, steps) + 1):
            x_adv = (x_orig + delta).clamp(-1.0, 1.0)
            x_eval = self.apply_eot_transform(x_adv) if enable_eot else x_adv

            adv_lat_sd, adv_feats_sd = self._extract_encoder_features(self.taesd, x_eval)

            # Maximize divergence from original latent + collapse spatial SSIM
            lat_mse_sd = F.mse_loss(adv_lat_sd.float(), orig_lat_sd.float())
            lat_cos_sd = 1.0 - F.cosine_similarity(
                adv_lat_sd.float().flatten(1), orig_lat_sd.float().flatten(1), dim=1
            ).mean()
            lat_ssim_loss_sd = 1.0 - _feature_ssim(adv_lat_sd, orig_lat_sd)

            layer_loss_sd = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for adv_f, orig_f in zip(adv_feats_sd, orig_feats_sd):
                layer_loss_sd = layer_loss_sd + F.mse_loss(adv_f.float(), orig_f.float()) + 0.5 * (
                    1.0 - _feature_ssim(adv_f, orig_f)
                )
            if adv_feats_sd:
                layer_loss_sd = layer_loss_sd / len(adv_feats_sd)

            joint_objective = lat_mse_sd + 0.75 * lat_cos_sd + 0.6 * lat_ssim_loss_sd + 0.8 * layer_loss_sd

            if self.taesdxl is not None and orig_lat_xl is not None:
                adv_lat_xl, adv_feats_xl = self._extract_encoder_features(self.taesdxl, x_eval)
                lat_mse_xl = F.mse_loss(adv_lat_xl.float(), orig_lat_xl.float())
                lat_ssim_loss_xl = 1.0 - _feature_ssim(adv_lat_xl, orig_lat_xl)
                layer_loss_xl = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                for adv_f, orig_f in zip(adv_feats_xl, orig_feats_xl):
                    layer_loss_xl = layer_loss_xl + F.mse_loss(adv_f.float(), orig_f.float())
                if adv_feats_xl:
                    layer_loss_xl = layer_loss_xl / len(adv_feats_xl)
                joint_objective = joint_objective + 0.85 * (lat_mse_xl + 0.5 * lat_ssim_loss_xl + 0.6 * layer_loss_xl)

            joint_objective.backward()

            with torch.no_grad():
                grad = delta.grad.float()
                g = grad / grad.abs().mean().clamp(min=1e-8)

                m_buf.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v_buf.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m_buf / (1.0 - beta1 ** step)
                v_hat = v_buf / (1.0 - beta2 ** step)

                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(1, steps)))
                step_lr = lr_min + (lr_max - lr_min) * cosine_decay

                updated = delta.data.float() + step_lr * (m_hat / (v_hat.sqrt() + adam_eps))
                # Hard L-infinity projection clamp
                delta.data = torch.clamp(updated.to(dtype=self.dtype), min=-hard_bound, max=hard_bound)
                delta.data = (x_orig + delta.data).clamp(-1.0, 1.0) - x_orig
                delta.grad.zero_()

        with torch.no_grad():
            x_protected = (x_orig + delta).clamp(-1.0, 1.0)
            if input_range == (0.0, 1.0):
                x_protected = (x_protected + 1.0) * 0.5

        return x_protected.to(device=orig_device, dtype=orig_dtype)

    def protect_pil_image(
        self,
        img: Image.Image,
        epsilon: float = DEFAULT_EPSILON,
        steps: int = DEFAULT_STEPS,
        enable_eot: bool = True,
        max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Preprocesses a PIL Image (strips metadata, bounds long edge, aligns to 8px),
        runs `protect_image_tensor`, and returns a clean 32-bit RGBA PIL Image + metrics.
        """
        t0 = time.perf_counter()
        clean_img = ImageOps.exif_transpose(img).convert("RGB")

        w, h = clean_img.size
        longest = max(w, h)
        if longest > max_long_edge:
            scale = max_long_edge / longest
            clean_img = clean_img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.LANCZOS,
            )

        aligned_w = max(8, (clean_img.width // 8) * 8)
        aligned_h = max(8, (clean_img.height // 8) * 8)
        if (aligned_w, aligned_h) != clean_img.size:
            clean_img = clean_img.resize((aligned_w, aligned_h), Image.Resampling.LANCZOS)

        arr = np.asarray(clean_img, dtype=np.float32) / 255.0
        x_in = (torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(
            device=self.device, dtype=self.dtype
        )

        x_out = self.protect_image_tensor(
            x_in,
            epsilon=epsilon,
            steps=steps,
            enable_eot=enable_eot,
            input_range=(-1.0, 1.0),
        )

        out_arr = ((x_out.detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round()
        out_arr = out_arr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        protected_rgba = Image.fromarray(out_arr, mode="RGB").convert("RGBA")

        psnr_db, ssim_val = compute_psnr_ssim(clean_img, protected_rgba)
        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

        return protected_rgba, {
            "width": aligned_w,
            "height": aligned_h,
            "epsilon": round(float(epsilon), 6),
            "steps": int(steps),
            "psnr": psnr_db,
            "ssim": ssim_val,
            "elapsed_ms": elapsed_ms,
        }

    def protect_image_bytes(
        self,
        raw_bytes: bytes | io.BytesIO,
        epsilon: float = DEFAULT_EPSILON,
        steps: int = DEFAULT_STEPS,
        enable_eot: bool = True,
        max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    ) -> tuple[io.BytesIO, dict[str, Any]]:
        """
        Zero-retention byte-stream wrapper: decodes incoming bytes in memory,
        immunizes pixels via `protect_pil_image`, and returns a lossless 32-bit PNG
        `io.BytesIO` buffer (`compress_level=0`) + quality metrics.
        """
        stream = raw_bytes if isinstance(raw_bytes, io.BytesIO) else io.BytesIO(raw_bytes)
        with Image.open(stream) as opened:
            protected_rgba, metrics = self.protect_pil_image(
                opened,
                epsilon=epsilon,
                steps=steps,
                enable_eot=enable_eot,
                max_long_edge=max_long_edge,
            )

        png_buf = encode_lossless_png_stream(protected_rgba)
        return png_buf, metrics


def protect_image_tensor(
    image_tensor: torch.Tensor,
    epsilon: float = DEFAULT_EPSILON,
    steps: int = DEFAULT_STEPS,
    enable_eot: bool = True,
    engine: PhotoGuardEngine | None = None,
) -> torch.Tensor:
    """
    Standalone functional entry point matching `protect_image_tensor(image_tensor, epsilon, steps)`.
    Lazily initializes a cached FP16 PhotoGuardEngine if one is not provided.
    """
    active_engine = engine or _get_default_engine()
    return active_engine.protect_image_tensor(
        image_tensor=image_tensor,
        epsilon=epsilon,
        steps=steps,
        enable_eot=enable_eot,
    )


_CACHED_ENGINE: PhotoGuardEngine | None = None


def _get_default_engine() -> PhotoGuardEngine:
    global _CACHED_ENGINE
    if _CACHED_ENGINE is None:
        _CACHED_ENGINE = PhotoGuardEngine.from_pretrained()
    return _CACHED_ENGINE
