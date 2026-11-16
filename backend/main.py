"""
PhotoGuard Lite — FastAPI Service
=================================
Zero-retention web API wired directly to `photoguard_engine.PhotoGuardEngine`.
"""

import re
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from photoguard_engine import DEFAULT_EPSILON, DEFAULT_STEPS, PhotoGuardEngine

PRESETS = {
    "light": {"epsilon": 4.0 / 255.0, "steps": 20},
    "balanced": {"epsilon": DEFAULT_EPSILON, "steps": DEFAULT_STEPS},
    "maximum": {"epsilon": DEFAULT_EPSILON, "steps": DEFAULT_STEPS},
}

app_state: dict[str, PhotoGuardEngine | None] = {"engine": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["engine"] = PhotoGuardEngine.from_pretrained()
    yield
    app_state["engine"] = None


app = FastAPI(
    title="PhotoGuard Lite API",
    description="Zero-retention EOT + Cross-Model Adam-PGD adversarial immunization service.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=[
        "X-Process-Time-Ms",
        "X-PSNR",
        "X-SSIM",
        "X-Epsilon",
        "X-Steps",
        "X-Format",
    ],
)


@app.get("/api/health")
async def health():
    engine = app_state.get("engine")
    if engine is None:
        return {"status": "initializing", "zero_retention": True}
    return {
        "status": "ok",
        "device": engine.device,
        "dtype": str(engine.dtype),
        "models": {
            "taesd": engine.taesd is not None,
            "taesdxl": engine.taesdxl is not None,
        },
        "zero_retention": True,
    }


def build_protected_filename(original_name: str | None) -> str:
    if not original_name:
        return "protected_image.png"
    base = re.sub(r"\.[^/.]+$", "", original_name)
    base = re.sub(r"^(protected_)+", "", base, flags=re.IGNORECASE)
    clean_base = re.sub(r"[^a-zA-Z0-9_-]", "_", base).strip("_") or "image"
    return f"protected_{clean_base}.png"


@app.post("/api/protect")
async def protect_image(
    file: UploadFile = File(...),
    strength: Literal["light", "balanced", "maximum"] = Form("balanced"),
    epsilon: float | None = Form(None),
    steps: int | None = Form(None),
    enable_eot: bool = Form(True),
):
    engine = app_state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="PhotoGuardEngine is not initialized.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload.")

    preset = PRESETS[strength]
    eps_norm = min(DEFAULT_EPSILON, float(epsilon)) if epsilon is not None else preset["epsilon"]
    num_steps = int(steps) if steps is not None else preset["steps"]

    try:
        png_buf, metrics = engine.protect_image_bytes(
            raw_bytes=raw_bytes,
            epsilon=eps_norm,
            steps=num_steps,
            enable_eot=enable_eot,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image immunization failed: {exc}") from exc

    safe_filename = build_protected_filename(file.filename)

    return StreamingResponse(
        png_buf,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "X-Process-Time-Ms": str(metrics["elapsed_ms"]),
            "X-PSNR": str(metrics["psnr"]),
            "X-SSIM": str(metrics["ssim"]),
            "X-Epsilon": str(metrics["epsilon"]),
            "X-Steps": str(metrics["steps"]),
            "X-Format": "PNG-32-LOSSLESS",
        },
    )
