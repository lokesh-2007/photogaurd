"""
PhotoGuard Lite — Cross-Attention & EOT Adversarial Engine
==========================================================
Upgrades the PGD optimization loop to jointly disrupt:
1. VAE latent encoding (pushing spatial representations toward collapse).
2. Diffusion UNet cross-attention maps (maximizing Shannon entropy across
   both token and spatial dimensions via explicit attention-processor hooks
   so text prompts like "face", "name", or "background" cannot bind to pixels).
3. Platform preprocessing invariance via an expanded differentiable EOT
   pipeline (random brightness/contrast shifts, micro-scaling, and
   differentiable 8x8 DCT JPEG compression simulation at quality 85-95).
"""

import io
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


DEFAULT_TARGET_PROMPT = (
    "a portrait photo of a person face, eyes, hair, skin, "
    "certificate text, full name, signature, logo, document background"
)


# Standard ITU-T.81 Luminance (Y) and Chrominance (Cb/Cr) 8x8 Quantization Tables
_JPEG_LUMA_TABLE = torch.tensor(
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

_JPEG_CHROMA_TABLE = torch.tensor(
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


def _build_dct_matrix(n: int = 8) -> torch.Tensor:
    """Builds the orthonormal 8x8 Type-II DCT basis matrix."""
    k = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    alpha = torch.ones((n, 1), dtype=torch.float32) * math.sqrt(2.0 / n)
    alpha[0, 0] = math.sqrt(1.0 / n)
    return alpha * torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))


class DifferentiableEOTModule(nn.Module):
    """
    Expectation Over Transformation (EOT) module executed every PGD iteration:
      - Random brightness & contrast perturbation (color/contrast invariance)
      - Random spatial micro-scaling (96% - 104%)
      - Differentiable 8x8 DCT JPEG compression simulation (quality 85 - 95)
    """

    def __init__(self, jpeg_quality_range=(85.0, 95.0), brightness_range=(-0.08, 0.08), contrast_range=(0.88, 1.12)):
        super().__init__()
        self.jpeg_quality_range = jpeg_quality_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        self.register_buffer("dct_mat", _build_dct_matrix(8))
        self.register_buffer("q_luma", _JPEG_LUMA_TABLE)
        self.register_buffer("q_chroma", _JPEG_CHROMA_TABLE)

    @staticmethod
    def _quality_to_scale(quality: float) -> float:
        q = max(1.0, min(100.0, float(quality)))
        if q < 50.0:
            return 5000.0 / q
        return 200.0 - 2.0 * q

    def _ste_quantize_plane(self, plane_255: torch.Tensor, q_table: torch.Tensor, scale: float) -> torch.Tensor:
        """Applies blockwise 8x8 DCT -> STE JPEG quantization -> 8x8 IDCT on a single [B, 1, H, W] plane."""
        b, _, h, w = plane_255.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h or pad_w:
            plane_padded = F.pad(plane_255, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            plane_padded = plane_255

        _, _, hp, wp = plane_padded.shape
        # Center around 128 per ITU-T.81
        centered = plane_padded - 128.0
        blocks = centered.view(b, hp // 8, 8, wp // 8, 8).permute(0, 1, 3, 2, 4).contiguous()

        c = self.dct_mat
        dct = torch.matmul(torch.matmul(c, blocks), c.t())

        q_scaled = torch.clamp((q_table * scale + 50.0) / 100.0, min=1.0, max=255.0)
        # Straight-Through Estimator: exact quantized values forward, smooth gradient backward
        quantized = torch.round(dct / q_scaled) * q_scaled
        dct_ste = dct + (quantized - dct).detach()

        idct = torch.matmul(torch.matmul(c.t(), dct_ste), c) + 128.0
        reconstructed = idct.permute(0, 1, 3, 2, 4).contiguous().view(b, 1, hp, wp)
        return reconstructed[:, :, :h, :w]

    def _simulate_jpeg(self, x: torch.Tensor, quality: float) -> torch.Tensor:
        """Differentiable RGB -> YCbCr -> 8x8 DCT JPEG (Q=85..95) -> RGB."""
        x_f32 = x.float() * 255.0
        r, g, b = x_f32[:, 0:1], x_f32[:, 1:2], x_f32[:, 2:3]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

        scale = self._quality_to_scale(quality)
        y_q = self._ste_quantize_plane(y, self.q_luma, scale)
        cb_q = self._ste_quantize_plane(cb, self.q_chroma, scale)
        cr_q = self._ste_quantize_plane(cr, self.q_chroma, scale)

        cb_s = cb_q - 128.0
        cr_s = cr_q - 128.0
        r_out = y_q + 1.402 * cr_s
        g_out = y_q - 0.344136 * cb_s - 0.714136 * cr_s
        b_out = y_q + 1.772 * cb_s

        rgb = torch.cat([r_out, g_out, b_out], dim=1) / 255.0
        return torch.clamp(rgb, 0.0, 1.0).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        # 1. Color & Contrast Invariance: random brightness shift + contrast scaling
        b_low, b_high = self.brightness_range
        c_low, c_high = self.contrast_range
        brightness = torch.empty(1, device=x.device, dtype=torch.float32).uniform_(b_low, b_high).item()
        contrast = torch.empty(1, device=x.device, dtype=torch.float32).uniform_(c_low, c_high).item()

        out_f32 = out.float()
        mean_val = out_f32.mean(dim=(-2, -1), keepdim=True)
        out_f32 = (out_f32 - mean_val) * contrast + mean_val + brightness
        out = torch.clamp(out_f32, 0.0, 1.0).to(dtype=x.dtype)

        # 2. Mild random spatial scaling (96% - 104%) to prevent pixel-grid overfitting
        scale_factor = torch.empty(1, device=x.device).uniform_(0.96, 1.04).item()
        if abs(scale_factor - 1.0) > 0.005:
            _, _, h, w = out.shape
            scaled_h = max(8, int(round(h * scale_factor)))
            scaled_w = max(8, int(round(w * scale_factor)))
            resized = F.interpolate(out.float(), size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
            out = F.interpolate(resized, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0).to(dtype=x.dtype)

        # 3. Differentiable JPEG compression simulation at quality 85 - 95
        q_low, q_high = self.jpeg_quality_range
        jpeg_quality = torch.empty(1, device=x.device).uniform_(q_low, q_high).item()
        out = self._simulate_jpeg(out, jpeg_quality)

        return out


class CrossAttentionMapCollector:
    """Collects softmax cross-attention probability maps captured during a UNet forward pass."""

    def __init__(self):
        self.maps: List[torch.Tensor] = []

    def clear(self):
        self.maps.clear()


class CrossAttentionEntropyProcessor:
    """
    Custom Diffusers attention processor for UNet cross-attention (`attn2`) layers.
    Replaces opaque FlashAttention during optimization so explicit softmax attention
    weights P = softmax(Q @ K^T * scale) remain differentiable and exposed for
    entropy maximization.
    """

    def __init__(self, collector: CrossAttentionMapCollector):
        self.collector = collector

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute explicit softmax attention probabilities in float32 for stable entropy gradients
        attn_scores = torch.bmm(query.float(), key.float().transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.float()

        attn_probs = torch.softmax(attn_scores, dim=-1)
        if is_cross:
            self.collector.maps.append(attn_probs)

        hidden_states = torch.bmm(attn_probs.to(value.dtype), value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


class CrossAttentionEntropyHook:
    """Context manager that attaches CrossAttentionEntropyProcessor to all UNet `attn2` modules."""

    def __init__(self, unet_model):
        self.unet = unet_model
        self.collector = CrossAttentionMapCollector()
        self.original_processors = None

    def __enter__(self) -> CrossAttentionMapCollector:
        if self.unet is None or not hasattr(self.unet, "attn_processors"):
            return self.collector

        self.original_processors = dict(self.unet.attn_processors)
        patched_processors = {}
        for name, proc in self.original_processors.items():
            # Target text-conditioning cross-attention layers (`attn2`)
            if "attn2" in name:
                patched_processors[name] = CrossAttentionEntropyProcessor(self.collector)
            else:
                patched_processors[name] = proc
        self.unet.set_attn_processor(patched_processors)
        return self.collector

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.unet is not None and self.original_processors is not None:
            self.unet.set_attn_processor(self.original_processors)
        self.collector.clear()


def compute_cross_attention_entropy_loss(attn_maps: List[torch.Tensor]) -> torch.Tensor:
    """
    Computes normalized Shannon entropy across UNet cross-attention maps.
    Maximizing this score destroys spatial binding between text prompt tokens
    ("face", "name", "background") and specific image regions:
      - H_token: forces each image pixel to attend uniformly over all tokens.
      - H_spatial: forces each prompt token to disperse uniformly across all pixels.
    """
    if not attn_maps:
        return torch.tensor(0.0)

    eps = 1e-7
    layer_entropies = []

    for probs in attn_maps:
        # probs: (B * heads, N_spatial, N_tokens)
        n_spatial = probs.shape[1]
        n_tokens = probs.shape[2]
        if n_spatial <= 1 or n_tokens <= 1:
            continue

        # 1. Per-pixel token entropy across N_tokens
        p_tok = torch.clamp(probs, min=eps, max=1.0)
        h_token = -torch.sum(p_tok * torch.log(p_tok), dim=-1) / math.log(n_tokens)

        # 2. Per-token spatial dispersion entropy across N_spatial
        spatial_dist = probs / (probs.sum(dim=1, keepdim=True) + eps)
        p_spa = torch.clamp(spatial_dist, min=eps, max=1.0)
        h_spatial = -torch.sum(p_spa * torch.log(p_spa), dim=1) / math.log(n_spatial)

        layer_entropies.append(0.5 * h_token.mean() + 0.5 * h_spatial.mean())

    if not layer_entropies:
        return torch.tensor(0.0, device=attn_maps[0].device)

    return torch.stack(layer_entropies).mean()


class DeepDiffusionPhotoGuardEngine:
    def __init__(
        self,
        vae_model,
        unet_model=None,
        scheduler=None,
        text_embeddings: Optional[torch.Tensor] = None,
        device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    ):
        self.vae = vae_model.to(device)
        self.unet = unet_model.to(device) if unet_model is not None else None
        self.scheduler = scheduler
        self.device = device
        self.dtype = next(self.vae.parameters()).dtype
        self.eot = DifferentiableEOTModule().to(device=self.device)
        self.text_embeddings = text_embeddings

        self.vae.eval()
        if self.unet is not None:
            self.unet.eval()

        for param in self.vae.parameters():
            param.requires_grad = False
        if self.unet is not None:
            for param in self.unet.parameters():
                param.requires_grad = False

    def _get_conditioning_embeddings(self) -> torch.Tensor:
        """
        Returns non-zero text-conditioning embeddings (1, 77, cross_attention_dim).
        Non-zero embeddings are mandatory so UNet cross-attention keys K = W_k(E)
        have rich semantic variance and non-vanishing attention entropy gradients.
        """
        cross_dim = getattr(getattr(self.unet, "config", None), "cross_attention_dim", 768)
        if self.text_embeddings is not None:
            return self.text_embeddings.to(device=self.device, dtype=self.dtype)

        gen = torch.Generator(device="cpu").manual_seed(42)
        emb = torch.randn((1, 77, cross_dim), generator=gen, dtype=torch.float32)
        # Match typical CLIP hidden-state token norms (~16.0)
        emb = F.normalize(emb, dim=-1) * 16.0
        return emb.to(device=self.device, dtype=self.dtype)

    def optimize_complex_perturbation(
        self,
        image_tensor: torch.Tensor,
        epsilon: float = 12.0 / 255.0,
        steps: int = 50,
        step_size: float = 2.0 / 255.0,
        attn_entropy_weight: float = 1.75,
        enable_eot: bool = True,
    ) -> torch.Tensor:
        """
        Executes a multi-objective momentum-PGD attack maximizing:
          1. VAE latent collapse penalty (pushing latents away from semantic content)
          2. UNet cross-attention Shannon entropy (breaking prompt-to-pixel binding)
          3. UNet denoising residual divergence
        while passing adversarial candidates through an EOT pipeline (random brightness,
        contrast, micro-scale, and differentiable JPEG Q=85..95 compression) every step.
        """
        x_orig = image_tensor.clone().detach().to(device=self.device, dtype=self.dtype)

        # Initialize inside the L-infinity epsilon ball
        x_adv = x_orig.clone().detach() + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv.clamp_(0.0, 1.0)

        with torch.no_grad():
            encoded_orig = self.vae.encode(x_orig * 2.0 - 1.0)
            base_latent = (
                encoded_orig.latent_dist.mean
                if hasattr(encoded_orig, "latent_dist")
                else encoded_orig.latents
            )
            target_latent = torch.zeros_like(base_latent)

        cond_embeddings = self._get_conditioning_embeddings() if self.unet is not None else None
        vae_scaling = getattr(getattr(self.vae, "config", None), "scaling_factor", 0.18215)

        momentum = torch.zeros_like(x_adv)
        mu = 0.9

        with CrossAttentionEntropyHook(self.unet) as attn_collector:
            for step_idx in range(steps):
                x_adv.requires_grad_(True)
                attn_collector.clear()

                # Apply EOT (brightness/contrast jitter + micro-scale + JPEG Q=85..95)
                x_transformed = self.eot(x_adv) if enable_eot else x_adv

                # Scale transformed pixels to [-1, 1] for VAE encoding
                encoded = self.vae.encode(x_transformed * 2.0 - 1.0)
                latents = (
                    encoded.latent_dist.mean
                    if hasattr(encoded, "latent_dist")
                    else encoded.latents
                )

                # Primary VAE latent collapse penalty (maximize distance from original representation)
                loss_vae_collapse = F.mse_loss(latents.float(), base_latent.float()) - 0.5 * F.mse_loss(
                    latents.float(), target_latent.float()
                )
                total_loss = loss_vae_collapse

                # Secondary UNet Cross-Attention Entropy + Noise Prediction Disruption
                if self.unet is not None and self.scheduler is not None and cond_embeddings is not None:
                    scaled_latents = latents * vae_scaling
                    t = torch.randint(15, 450, (1,), device=self.device).long()
                    noise = torch.randn_like(scaled_latents)
                    noisy_latents = self.scheduler.add_noise(scaled_latents, noise, t)

                    noise_pred = self.unet(
                        noisy_latents,
                        t,
                        encoder_hidden_states=cond_embeddings,
                    ).sample

                    # Maximize Shannon entropy in captured cross-attention maps (attn2 layers)
                    loss_attn_entropy = compute_cross_attention_entropy_loss(attn_collector.maps)
                    loss_noise_div = F.mse_loss(noise_pred.float(), noise.float())

                    total_loss = total_loss + attn_entropy_weight * loss_attn_entropy + 0.35 * loss_noise_div

                # Compute gradients w.r.t. raw adversarial tensor x_adv
                grad = torch.autograd.grad(total_loss, [x_adv], retain_graph=False)[0]

                # Momentum-stabilized gradient ascent (MI-FGSM)
                grad_norm = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
                momentum = mu * momentum + grad_norm

                # Cosine decay on step size for clean high-frequency convergence
                decay = 0.5 * (1.0 + math.cos(math.pi * step_idx / max(1, steps)))
                cur_step = max(step_size * 0.25, step_size * (0.4 + 0.6 * decay))

                x_adv = x_adv + cur_step * torch.sign(momentum)

                # Strict L-infinity projection and [0, 1] pixel clipping
                eta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
                x_adv = torch.clamp(x_orig + eta, min=0.0, max=1.0).detach()

        return x_adv


def protect_image_stream_v2(
    image_bytes: bytes,
    vae_model,
    unet_model=None,
    scheduler=None,
    epsilon_val: float = 12.0 / 255.0,
    text_embeddings: Optional[torch.Tensor] = None,
    steps: int = 50,
    enable_eot: bool = True,
) -> bytes:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    engine = DeepDiffusionPhotoGuardEngine(
        vae_model=vae_model,
        unet_model=unet_model,
        scheduler=scheduler,
        text_embeddings=text_embeddings,
        device=device,
    )

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_img = transforms.ToTensor()(pil_img).unsqueeze(0)

    protected_tensor = engine.optimize_complex_perturbation(
        tensor_img,
        epsilon=epsilon_val,
        steps=steps,
        enable_eot=enable_eot,
    )

    protected_pil = transforms.ToPILImage()(protected_tensor.squeeze(0).float().cpu()).convert("RGBA")

    output_buffer = io.BytesIO()
    protected_pil.save(output_buffer, format="PNG", compress_level=0)
    output_buffer.seek(0)

    return output_buffer.getvalue()
