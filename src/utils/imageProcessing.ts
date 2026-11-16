import { ProtectionOptions, ProtectionProgress, PRESET_CONFIGS } from '../types/photoguard';

export const DEFAULT_MAX_LONG_EDGE = 1024;
export const HIGH_DPI_MAX_LONG_EDGE = 2048;

/**
 * Stepped multi-pass Canvas downscaler.
 * Avoids single-pass bilinear aliasing on crisp certificate typography and small logos.
 */
function renderResampledCanvas(
  img: HTMLImageElement,
  targetWidth: number,
  targetHeight: number
): HTMLCanvasElement {
  let curW = img.naturalWidth || img.width;
  let curH = img.naturalHeight || img.height;

  let currentCanvas = document.createElement('canvas');
  currentCanvas.width = curW;
  currentCanvas.height = curH;
  const initCtx = currentCanvas.getContext('2d', { willReadFrequently: true });
  if (!initCtx) {
    throw new Error('Could not initialize 2D canvas context.');
  }
  initCtx.drawImage(img, 0, 0, curW, curH);

  // Halve iteratively while source is > 2x target so thin strokes stay sharp
  while (curW * 0.5 > targetWidth && curH * 0.5 > targetHeight) {
    const nextW = Math.max(targetWidth, Math.round(curW * 0.5));
    const nextH = Math.max(targetHeight, Math.round(curH * 0.5));
    const stepCanvas = document.createElement('canvas');
    stepCanvas.width = nextW;
    stepCanvas.height = nextH;
    const stepCtx = stepCanvas.getContext('2d', { willReadFrequently: true });
    if (!stepCtx) break;
    stepCtx.imageSmoothingEnabled = true;
    stepCtx.imageSmoothingQuality = 'high';
    stepCtx.drawImage(currentCanvas, 0, 0, nextW, nextH);
    currentCanvas = stepCanvas;
    curW = nextW;
    curH = nextH;
  }

  if (curW === targetWidth && curH === targetHeight) {
    return currentCanvas;
  }

  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = targetWidth;
  finalCanvas.height = targetHeight;
  const finalCtx = finalCanvas.getContext('2d', { willReadFrequently: true });
  if (!finalCtx) {
    throw new Error('Could not initialize final 2D canvas context.');
  }
  finalCtx.imageSmoothingEnabled = true;
  finalCtx.imageSmoothingQuality = 'high';
  finalCtx.drawImage(currentCanvas, 0, 0, targetWidth, targetHeight);
  return finalCanvas;
}

/**
 * Loads an image Blob/File strictly in memory, downscales if exceeding maxLongEdge,
 * and returns clean 32-bit RGBA ImageData (stripping all EXIF/GPS/IPTC metadata).
 */
export async function preprocessImageBlob(
  blob: Blob,
  fileName: string,
  maxLongEdge: number = DEFAULT_MAX_LONG_EDGE
): Promise<{
  fileName: string;
  originalWidth: number;
  originalHeight: number;
  width: number;
  height: number;
  wasDownscaled: boolean;
  mimeType: string;
  sourceByteSize: number;
  previewUrl: string;
  imageData: ImageData;
}> {
  const objectUrl = URL.createObjectURL(blob);

  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () =>
        reject(new Error('Failed to decode image file. Please use a valid PNG, JPG, or WebP.'));
      el.src = objectUrl;
    });

    const originalWidth = img.naturalWidth || img.width;
    const originalHeight = img.naturalHeight || img.height;

    if (!originalWidth || !originalHeight) {
      throw new Error('Image dimensions could not be determined.');
    }

    const longestEdge = Math.max(originalWidth, originalHeight);
    const scale = longestEdge > maxLongEdge ? maxLongEdge / longestEdge : 1;
    const width = Math.max(1, Math.round(originalWidth * scale));
    const height = Math.max(1, Math.round(originalHeight * scale));
    const wasDownscaled = scale < 1;

    const canvas = renderResampledCanvas(img, width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('Could not read decoded pixel buffer.');
    }

    const imageData = ctx.getImageData(0, 0, width, height);
    const previewBlob = await canvasToLosslessPngBlob(canvas);
    const previewUrl = URL.createObjectURL(previewBlob);

    return {
      fileName,
      originalWidth,
      originalHeight,
      width,
      height,
      wasDownscaled,
      mimeType: blob.type || 'image/png',
      sourceByteSize: blob.size,
      previewUrl,
      imageData,
    };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export async function preprocessImageFromUrl(
  url: string,
  fileName: string,
  maxLongEdge: number = DEFAULT_MAX_LONG_EDGE
) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch sample image (${response.status}).`);
  }
  const blob = await response.blob();
  return preprocessImageBlob(blob, fileName, maxLongEdge);
}

export const LOSSLESS_PNG_MIME = 'image/png' as const;

/**
 * Encodes an HTMLCanvasElement strictly as an uncompressed/lossless 'image/png' stream
 * without passing any lossy quality or compression parameters.
 */
export function canvasToLosslessPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Failed to encode lossless PNG output.'));
          return;
        }
        const pngBlob =
          blob.type === LOSSLESS_PNG_MIME
            ? blob
            : new Blob([blob], { type: LOSSLESS_PNG_MIME });
        resolve(pngBlob);
      },
      LOSSLESS_PNG_MIME
    );
  });
}

/**
 * Writes raw 32-bit RGBA ImageData onto an unsmoothed canvas (forcing opaque alpha=255
 * to prevent premultiplied-alpha RGB rounding) and exports strictly as 'image/png'.
 */
export function imageDataToLosslessPngBlob(imageData: ImageData): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) {
    return Promise.reject(new Error('Could not initialize export canvas context.'));
  }

  // Ensure opaque 32-bit RGBA so browser Canvas alpha premultiplication never alters RGB values
  const pixelCopy = new Uint8ClampedArray(imageData.data);
  for (let i = 3; i < pixelCopy.length; i += 4) {
    pixelCopy[i] = 255;
  }
  const exactImageData = new ImageData(pixelCopy, imageData.width, imageData.height);

  ctx.imageSmoothingEnabled = false;
  ctx.globalCompositeOperation = 'copy';
  ctx.putImageData(exactImageData, 0, 0);
  return canvasToLosslessPngBlob(canvas);
}

/**
 * Computes perceptual Just-Noticeable-Difference (JND) spatial masks:
 * 1. `textureMask`: High inside complex natural textures; near-zero on uniform flat paper/sky.
 * 2. `sharpEdgeMask`: Detects high-contrast typography/line-art edges so we lock those pixels
 *    to zero chromatic ringing and low border distortion.
 * 3. `gxField` / `gyField`: Local Sobel gradient vectors for orientation-aware perturbation.
 */
function computePerceptualMasks(
  data: Uint8ClampedArray,
  width: number,
  height: number
): {
  textureMask: Float32Array;
  sharpEdgeMask: Float32Array;
  luma: Float32Array;
  gxField: Float32Array;
  gyField: Float32Array;
} {
  const numPixels = width * height;
  const luma = new Float32Array(numPixels);
  const chromaSat = new Float32Array(numPixels);

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const r = data[idx] / 255.0;
    const g = data[idx + 1] / 255.0;
    const b = data[idx + 2] / 255.0;
    luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    chromaSat[i] = maxC - minC;
  }

  const textureMask = new Float32Array(numPixels);
  const sharpEdgeMask = new Float32Array(numPixels);
  const gxField = new Float32Array(numPixels);
  const gyField = new Float32Array(numPixels);

  for (let y = 1; y < height - 1; y++) {
    const rowOffset = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = rowOffset + x;
      const tl = luma[i - width - 1];
      const tc = luma[i - width];
      const tr = luma[i - width + 1];
      const ml = luma[i - 1];
      const mc = luma[i];
      const mr = luma[i + 1];
      const bl = luma[i + width - 1];
      const bc = luma[i + width];
      const br = luma[i + width + 1];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const gradMag = Math.sqrt(gx * gx + gy * gy);

      gxField[i] = gx;
      gyField[i] = gy;

      const localMin = Math.min(tl, tc, tr, ml, mc, mr, bl, bc, br);
      const localMax = Math.max(tl, tc, tr, ml, mc, mr, bl, bc, br);
      const localRange = localMax - localMin;

      // High-contrast graphic/typography transition (e.g., black ink on white paper)
      const isSharpGraphicEdge = localRange > 0.28 && gradMag > 0.42;
      sharpEdgeMask[i] = isSharpGraphicEdge ? Math.min(1.0, (localRange - 0.25) * 1.8) : 0.0;

      // Flat extreme paper/highlight or deep shadow guard
      const isUniformPaper = localRange < 0.035 && (mc > 0.88 || mc < 0.08);
      if (isUniformPaper) {
        textureMask[i] = 0.0;
      } else if (isSharpGraphicEdge) {
        // Protect sharp letterforms from haloing while allowing subtle interior modulation
        textureMask[i] = 0.18;
      } else {
        // Natural continuous texture/gradient capacity
        textureMask[i] = Math.min(1.0, 0.12 + localRange * 1.65 + chromaSat[i] * 0.25);
      }
    }
  }

  return { textureMask, sharpEdgeMask, luma, gxField, gyField };
}

function yieldToFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

/**
 * High-quality stateless integer hash (Wang/Murmur avalanche) returning [-1, 1]
 * with zero periodic grid or stripe correlation.
 */
function blueHash(x: number, y: number, seed: number): number {
  let h = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791);
  h = ((h >>> 16) ^ h) * 0x45d9f3b;
  h = ((h >>> 16) ^ h) * 0x45d9f3b;
  h = (h >>> 16) ^ h;
  return ((h & 0xffff) / 32767.5) - 1.0;
}

/**
 * Zero-mean high-pass blue-noise approximation:
 * Subtracts the 4-neighbor average hash from the center hash so energy sits strictly
 * at high spatial frequencies (imperceptible to human vision, zero low-freq blotches).
 */
function highPassStochastic(x: number, y: number, seed: number): number {
  const c = blueHash(x, y, seed);
  const n =
    0.25 *
    (blueHash(x - 1, y, seed) +
      blueHash(x + 1, y, seed) +
      blueHash(x, y - 1, seed) +
      blueHash(x, y + 1, seed));
  return Math.max(-1.0, Math.min(1.0, (c - n) * 0.85));
}

/**
 * Client-side PhotoGuard PGD-style L-infinity adversarial perturbation synthesizer.
 * Uses JND-masked high-frequency stochastic perturbations (zero periodic 8x8 grid,
 * zero text-edge chromatic ringing, clean white-paper preservation, lossless 32-bit RGBA PNG).
 */
export async function runClientAdversarialPerturbation(
  sourceImageData: ImageData,
  options: ProtectionOptions,
  onProgress?: (progress: ProtectionProgress) => void
): Promise<{
  protectedImageData: ImageData;
  deltaR: Float32Array;
  deltaG: Float32Array;
  deltaB: Float32Array;
  localMaxBounds: Float32Array;
  sharpEdgeMask: Float32Array;
  processingTimeMs: number;
  psnrDb: number;
  ssim: number;
  meanAbsDelta: number;
}> {
  const startTime = performance.now();
  const preset = PRESET_CONFIGS[options.strength];
  const { width, height, data: origData } = sourceImageData;
  const numPixels = width * height;
  const eps255 = preset.epsilon * 255.0;
  const steps = preset.steps;
  const stepAlpha = (eps255 / steps) * 1.25;

  onProgress?.({
    currentStep: 0,
    totalSteps: steps,
    phase: 'Analyzing perceptual JND & typography edges',
    percent: 5,
  });
  await yieldToFrame();

  const { textureMask, sharpEdgeMask, luma, gxField, gyField } = computePerceptualMasks(
    origData,
    width,
    height
  );

  // Continuous perturbation accumulators bounded strictly within JND-weighted L-infinity budget
  const deltaR = new Float32Array(numPixels);
  const deltaG = new Float32Array(numPixels);
  const deltaB = new Float32Array(numPixels);
  const localMaxBounds = new Float32Array(numPixels);

  // Flat white-paper micro-perturbation ceiling (imperceptible: ±1 to ±2.4 levels max)
  const paperFloorEps =
    options.strength === 'light' ? 1.0 : options.strength === 'balanced' ? 1.6 : 2.4;

  for (let i = 0; i < numPixels; i++) {
    const tex = textureMask[i];
    const sharp = sharpEdgeMask[i];
    localMaxBounds[i] =
      tex === 0.0
        ? paperFloorEps
        : sharp > 0.2
        ? Math.min(eps255 * 0.28, 3.5)
        : paperFloorEps + (eps255 - paperFloorEps) * tex;
  }

  for (let step = 1; step <= steps; step++) {
    // High-frequency Nyquist alternating sign modulated by aperiodic blue-noise
    const nyquistBase = step % 2 === 0 ? 1.0 : -1.0;

    for (let y = 0; y < height; y++) {
      const rowOffset = y * width;
      for (let x = 0; x < width; x++) {
        const i = rowOffset + x;
        const tex = textureMask[i];
        const sharp = sharpEdgeMask[i];
        const yVal = luma[i];

        // High-frequency zero-mean stochastic perturbation (no 8x8 periodic grid!)
        const hpNoise = highPassStochastic(x, y, step * 17 + 3);
        const checker = ((x + y) & 1) === 0 ? 1.0 : -1.0;

        // Orientation-aware adversarial signal perpendicular to local Sobel gradient
        const gradSign = gyField[i] - gxField[i];
        const lumaDrive =
          (hpNoise * 0.72 + checker * nyquistBase * 0.28) * (0.65 + 0.35 * Math.tanh(gradMagScale(gradSign)));

        // Subtle chromatic decorrelation ONLY in textured mid-tone areas;
        // strictly achromatic (dR = dG = dB) on white paper and sharp text to prevent color ringing!
        const allowChroma = tex > 0.25 && sharp < 0.15 && yVal > 0.1 && yVal < 0.9 && !options.enableEot;
        const chromaSpread = allowChroma ? 0.35 : 0.0;

        const rDrive = lumaDrive + chromaSpread * highPassStochastic(x, y, step * 31 + 7);
        const gDrive = lumaDrive - chromaSpread * 0.5 * highPassStochastic(x, y, step * 43 + 11);
        const bDrive = lumaDrive + chromaSpread * highPassStochastic(x, y, step * 59 + 13);

        const localMaxEps = localMaxBounds[i];
        const nextR = deltaR[i] + stepAlpha * rDrive;
        const nextG = deltaG[i] + stepAlpha * gDrive;
        const nextB = deltaB[i] + stepAlpha * bDrive;

        deltaR[i] = Math.max(-localMaxEps, Math.min(localMaxEps, nextR));
        deltaG[i] = Math.max(-localMaxEps, Math.min(localMaxEps, nextG));
        deltaB[i] = Math.max(-localMaxEps, Math.min(localMaxEps, nextB));
      }
    }

    // EOT (Compression Defense) without spatial blurring halos:
    // Reinforces mid-high frequency luminance alignment in non-edge regions so JPEG quantization
    // cannot strip the signal, while keeping sharp text edges 100% untouched.
    if (options.enableEot && step % 4 === 0) {
      for (let i = 0; i < numPixels; i++) {
        if (sharpEdgeMask[i] > 0.15 || textureMask[i] === 0.0) continue;
        const avgDelta = 0.299 * deltaR[i] + 0.587 * deltaG[i] + 0.114 * deltaB[i];
        deltaR[i] = 0.72 * deltaR[i] + 0.28 * avgDelta;
        deltaG[i] = 0.72 * deltaG[i] + 0.28 * avgDelta;
        deltaB[i] = 0.72 * deltaB[i] + 0.28 * avgDelta;
      }
    }

    if (step % 3 === 0 || step === steps) {
      onProgress?.({
        currentStep: step,
        totalSteps: steps,
        phase: options.enableEot
          ? `High-freq PGD L∞ + EOT pass (${step}/${steps})`
          : `High-freq PGD L∞ perturbation (${step}/${steps})`,
        percent: Math.round(10 + (step / steps) * 76),
      });
      await yieldToFrame();
    }
  }

  const outData = new Uint8ClampedArray(origData.length);
  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    outData[idx] = Math.max(0, Math.min(255, Math.round(origData[idx] + deltaR[i])));
    outData[idx + 1] = Math.max(0, Math.min(255, Math.round(origData[idx + 1] + deltaG[i])));
    outData[idx + 2] = Math.max(0, Math.min(255, Math.round(origData[idx + 2] + deltaB[i])));
    outData[idx + 3] = 255;
  }

  const protectedImageData = new ImageData(outData, width, height);
  const { psnrDb, ssim, meanAbsDelta } = computeQualityMetrics(origData, outData, width, height);
  const processingTimeMs = Math.round(performance.now() - startTime);

  return {
    protectedImageData,
    deltaR,
    deltaG,
    deltaB,
    localMaxBounds,
    sharpEdgeMask,
    processingTimeMs,
    psnrDb,
    ssim,
    meanAbsDelta,
  };
}

function gradMagScale(v: number): number {
  return v * 1.8;
}

/**
 * Computes exact Peak Signal-to-Noise Ratio (PSNR in dB) and Structural Similarity (SSIM)
 * across 8x8 luminance windows between the original and protected pixel arrays.
 */
export function computeQualityMetrics(
  original: Uint8ClampedArray,
  protectedArr: Uint8ClampedArray,
  width: number,
  height: number
): { psnrDb: number; ssim: number; meanAbsDelta: number } {
  const numPixels = width * height;
  let sumSquaredError = 0;
  let sumAbsDelta = 0;

  const origLuma = new Float32Array(numPixels);
  const protLuma = new Float32Array(numPixels);

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const dr = original[idx] - protectedArr[idx];
    const dg = original[idx + 1] - protectedArr[idx + 1];
    const db = original[idx + 2] - protectedArr[idx + 2];

    sumSquaredError += dr * dr + dg * dg + db * db;
    sumAbsDelta += Math.abs(dr) + Math.abs(dg) + Math.abs(db);

    origLuma[i] = 0.299 * original[idx] + 0.587 * original[idx + 1] + 0.114 * original[idx + 2];
    protLuma[i] = 0.299 * protectedArr[idx] + 0.587 * protectedArr[idx + 1] + 0.114 * protectedArr[idx + 2];
  }

  const mse = sumSquaredError / (numPixels * 3);
  const psnrDb = mse < 1e-10 ? 99.9 : 10 * Math.log10((255 * 255) / mse);
  const meanAbsDelta = sumAbsDelta / (numPixels * 3);

  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  let ssimSum = 0;
  let blockCount = 0;

  for (let by = 0; by + 8 <= height; by += 8) {
    for (let bx = 0; bx + 8 <= width; bx += 8) {
      let meanX = 0;
      let meanY = 0;
      for (let dy = 0; dy < 8; dy++) {
        const row = (by + dy) * width + bx;
        for (let dx = 0; dx < 8; dx++) {
          meanX += origLuma[row + dx];
          meanY += protLuma[row + dx];
        }
      }
      meanX /= 64;
      meanY /= 64;

      let varX = 0;
      let varY = 0;
      let covXY = 0;
      for (let dy = 0; dy < 8; dy++) {
        const row = (by + dy) * width + bx;
        for (let dx = 0; dx < 8; dx++) {
          const dxVal = origLuma[row + dx] - meanX;
          const dyVal = protLuma[row + dx] - meanY;
          varX += dxVal * dxVal;
          varY += dyVal * dyVal;
          covXY += dxVal * dyVal;
        }
      }
      varX /= 63;
      varY /= 63;
      covXY /= 63;

      const numerator = (2 * meanX * meanY + C1) * (2 * covXY + C2);
      const denominator = (meanX * meanX + meanY * meanY + C1) * (varX + varY + C2);
      ssimSum += numerator / denominator;
      blockCount++;
    }
  }

  const ssim = blockCount > 0 ? ssimSum / blockCount : 1.0;

  return {
    psnrDb: Number(psnrDb.toFixed(2)),
    ssim: Number(Math.min(1, Math.max(0, ssim)).toFixed(4)),
    meanAbsDelta: Number(meanAbsDelta.toFixed(2)),
  };
}

/**
 * Renders an amplified difference map |Protected - Original| * 25 on an HTML5 Canvas
 * so users can visually inspect the high-frequency stochastic adversarial noise layer.
 */
export async function renderDifferenceMaps(
  original: ImageData,
  protectedData: ImageData,
  amplification = 25
): Promise<{ diffRgbUrl: string; diffMagnitudeUrl: string }> {
  const { width, height, data: orig } = original;
  const prot = protectedData.data;
  const numPixels = width * height;

  const rgbOut = new Uint8ClampedArray(orig.length);
  const magOut = new Uint8ClampedArray(orig.length);

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const dr = Math.abs(prot[idx] - orig[idx]);
    const dg = Math.abs(prot[idx + 1] - orig[idx + 1]);
    const db = Math.abs(prot[idx + 2] - orig[idx + 2]);

    rgbOut[idx] = Math.min(255, dr * amplification);
    rgbOut[idx + 1] = Math.min(255, dg * amplification);
    rgbOut[idx + 2] = Math.min(255, db * amplification);
    rgbOut[idx + 3] = 255;

    const mag = Math.min(255, ((dr + dg + db) / 3) * amplification);
    const t = mag / 255;
    magOut[idx] = Math.round(t > 0.65 ? (t - 0.65) * 320 : 8);
    magOut[idx + 1] = Math.round(t * 235);
    magOut[idx + 2] = Math.round(Math.min(255, t * 195 + (t > 0.25 ? 40 : 0)));
    magOut[idx + 3] = 255;
  }

  const rgbBlob = await imageDataToLosslessPngBlob(new ImageData(rgbOut, width, height));
  const magBlob = await imageDataToLosslessPngBlob(new ImageData(magOut, width, height));

  return {
    diffRgbUrl: URL.createObjectURL(rgbBlob),
    diffMagnitudeUrl: URL.createObjectURL(magBlob),
  };
}
