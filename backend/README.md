# PhotoGuard Lite — Lightweight MIT PhotoGuard FastAPI Backend

The AI Studio live preview runs an instant 100% client-side HTML5 Canvas adversarial perturbation engine with Floyd-Steinberg stochastic dithering. For GPU/CPU PyTorch inference, the backend extracts and refactors MIT MadryLab's PhotoGuard encoder attack into two clean files:

- **`backend/photoguard_engine.py`** — Low-VRAM (`<500 MB`, FP16 `TAESD` / `TAESDXL`) adversarial optimization engine exposing:
  - `protect_image_tensor(image_tensor, epsilon, steps)` — Pure tensor PGD optimization with Adam momentum, strict $L_\infty$ clamping (`ε ≤ 6/255`), cross-model latent/feature SSIM collapse, and EOT augmentations (random 95%–105% scaling, Gaussian blur, color jitter, and differentiable 8×8 DCT JPEG simulation).
  - `PhotoGuardEngine.protect_image_bytes(...)` — In-memory `BytesIO`/PIL wrapper that strips EXIF/GPS metadata and outputs a guaranteed lossless 32-bit RGBA PNG buffer (`compress_level=0`).
- **`backend/main.py`** — Streamlined FastAPI `/api/protect` server wired directly to `PhotoGuardEngine`.

## Quick Start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Connecting the React Frontend

1. Set `VITE_PHOTOGUARD_API_URL=http://localhost:8000` (or proxy `/api` in Vite).
2. In `src/services/immunizeService.ts`, pass `'api'` as `engineMode` to `immunizeImage(...)`.
