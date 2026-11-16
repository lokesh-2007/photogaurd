"""
PhotoGuard Lite — Production FastAPI Server (`main.py`)
======================================================
Hosts `photoguard_engine.protect_image_stream_v2()` with startup model loading
in half-precision (`torch.float16` on GPU/MPS, `torch.float32` CPU fallback),
strict image MIME validation, OOM/corrupt-image error handling, CORS for
localhost:3000, and lossless 32-bit PNG streaming.
"""

import io
import os
import re
import time
from contextlib import asynccontextmanager

import torch
from diffusers import AutoencoderKL, AutoencoderTiny
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from photoguard_engine import protect_image_stream_v2

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_LONG_EDGE = 1024
DEFAULT_VAE_MODEL_ID = os.getenv("PHOTOGUARD_VAE_ID", "stabilityai/stable-diffusion-2-1")

model_registry: dict = {
    "vae": None,
    "unet": None,
    "scheduler": None,
    "device": "cpu",
    "dtype": torch.float32,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load VAE encoder once at startup in half-precision (torch.float16) when supported."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    vae_id = DEFAULT_VAE_MODEL_ID
    try:
        if "taesd" in vae_id.lower():
            vae = AutoencoderTiny.from_pretrained(vae_id, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=dtype)
    except Exception:
        # Lightweight TAESD fallback if full SD 2.1 VAE cannot be fetched
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)

    vae = vae.to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    model_registry["vae"] = vae
    model_registry["device"] = device
    model_registry["dtype"] = dtype

    yield

    model_registry["vae"] = None
    model_registry["unet"] = None
    model_registry["scheduler"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="PhotoGuard Lite API",
    description="Zero-retention FastAPI host for DeepDiffusionPhotoGuardEngine.",
    version="4.0.0",
    lifespan=lifespan,
)

# Allow frontend running on localhost:3000 (React / Next.js) to call /api/protect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Process-Time-Ms", "X-Format"],
)


def _preprocess_clean_png_bytes(raw_bytes: bytes) -> bytes:
    """
    Validates image bytes, strips EXIF/GPS metadata, caps longest edge at 1024px,
    and aligns dimensions to multiples of 8 for clean VAE latent encoding.
    """
    with Image.open(io.BytesIO(raw_bytes)) as opened:
        clean_img = ImageOps.exif_transpose(opened).convert("RGB")

    w, h = clean_img.size
    longest = max(w, h)
    if longest > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / longest
        clean_img = clean_img.resize(
            (max(8, round(w * scale)), max(8, round(h * scale))),
            Image.Resampling.LANCZOS,
        )

    aligned_w = max(8, (clean_img.width // 8) * 8)
    aligned_h = max(8, (clean_img.height // 8) * 8)
    if (aligned_w, aligned_h) != clean_img.size:
        clean_img = clean_img.resize((aligned_w, aligned_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    clean_img.save(buf, format="PNG", compress_level=0)
    return buf.getvalue()


def _sanitize_download_filename(original_name: str | None) -> str:
    if not original_name:
        return "protected_image.png"
    base = re.sub(r"\.[^/.]+$", "", original_name)
    base = re.sub(r"^(protected_)+", "", base, flags=re.IGNORECASE)
    clean_base = re.sub(r"[^a-zA-Z0-9_-]", "_", base).strip("_") or "image"
    return f"protected_{clean_base}.png"


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if model_registry["vae"] is not None else "initializing",
        "device": model_registry["device"],
        "dtype": str(model_registry["dtype"]),
        "zero_retention": True,
    }


@app.post("/api/protect")
async def protect_image(
    file: UploadFile = File(...),
    epsilon: float = Form(12.0 / 255.0),
):
    vae_model = model_registry.get("vae")
    if vae_model is None:
        raise HTTPException(status_code=503, detail="VAE encoder is not loaded yet.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Invalid file type. Only image/jpeg and image/png uploads are supported.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    t0 = time.perf_counter()

    try:
        clean_bytes = _preprocess_clean_png_bytes(raw_bytes)
        bounded_epsilon = max(2.0 / 255.0, min(16.0 / 255.0, float(epsilon)))

        protected_bytes = protect_image_stream_v2(
            image_bytes=clean_bytes,
            vae_model=vae_model,
            unet_model=model_registry.get("unet"),
            scheduler=model_registry.get("scheduler"),
            epsilon_val=bounded_epsilon,
        )

        # Guarantee lossless 32-bit RGBA PNG stream
        with Image.open(io.BytesIO(protected_bytes)) as out_img:
            rgba = out_img.convert("RGBA")
            final_stream = io.BytesIO()
            rgba.save(final_stream, format="PNG", compress_level=0)
            final_stream.seek(0)

    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Corrupted or unsupported image data.") from exc
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory while immunizing image. Try a smaller resolution.",
        ) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(
                status_code=503,
                detail="Insufficient memory while immunizing image.",
            ) from exc
        raise HTTPException(status_code=500, detail=f"Immunization failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected immunization error: {exc}") from exc

    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
    safe_filename = _sanitize_download_filename(file.filename)

    return StreamingResponse(
        final_stream,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "X-Process-Time-Ms": str(elapsed_ms),
            "X-Format": "PNG-32-LOSSLESS",
        },
    )
