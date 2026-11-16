# PhotoGuard Lite — Multi-Scale Latent & UNet Maximization Backend

The AI Studio live preview runs an instant 100% client-side HTML5 Canvas adversarial perturbation engine with Floyd-Steinberg stochastic dithering. For GPU/CPU PyTorch inference, the backend implements the maximum-strength multi-layer defense in:

- **`backend/engine.py`** (`PhotoGuardMultiScaleEngine`):
  - **Multi-Layer Divergence Maximization:** Maximizes L2, cosine, and SSIM divergence simultaneously across VAE encoder latents (`madebyollin/taesd` & `taesdxl`), early VAE convolutional layers, and UNet early down-sampling blocks (`_UNetSpatialDownSurrogate` + optional full `UNet2DConditionModel` via `PHOTOGUARD_UNET_ID`).
  - **Adaptive Adam-Momentum PGD (40–50 steps):** Uses bias-corrected Adam moments (`β1=0.9, β2=0.999`) blended with signed gradient exploration and cosine-annealed step decay.
  - **Dynamic Perceptual Masking (`ε ≤ 8/255`):** Scales perturbation intensity dynamically up to `8/255` in textured/busy regions while clamping flat white backgrounds (`~2/255`) and sharp typography borders (`~3.5/255`).
  - **EOT + Lossless 32-Bit PNG:** Applies random `95%–105%` scaling, Gaussian blur, color jitter, and differentiable 8×8 DCT JPEG simulation inside each iteration; strips EXIF/GPS tags and outputs raw 32-bit RGBA PNG (`compress_level=0`).
- **`backend/main.py`**: Streamlined FastAPI `/api/protect` service wired directly to `engine.py`.

## Quick Start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
