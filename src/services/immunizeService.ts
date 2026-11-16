import {
  PRESET_CONFIGS,
  PreprocessedImage,
  ProtectionOptions,
  ProtectionProgress,
  ProtectionResult,
} from '../types/photoguard';
import {
  LOSSLESS_PNG_MIME,
  computeQualityMetrics,
  imageDataToLosslessPngBlob,
  preprocessImageBlob,
  renderDifferenceMaps,
  runClientAdversarialPerturbation,
} from '../utils/imageProcessing';

/**
 * Execution mode:
 * - 'client': Runs the pure in-browser HTML5 Canvas adversarial engine (instant preview).
 * - 'api': Calls the FastAPI PyTorch TAESD PGD backend at POST /api/protect.
 */
export type ProcessingEngineMode = 'client' | 'api';

export function formatProtectedPngFilename(originalName: string): string {
  const withoutExt = originalName.replace(/\.[^/.]+$/, '');
  const withoutPrefix = withoutExt.replace(/^(protected_)+/i, '');
  const cleanBase = withoutPrefix.replace(/[^a-zA-Z0-9_-]/g, '_');
  return `protected_${cleanBase || 'image'}.png`;
}

function buildPngSourceFilename(originalName: string): string {
  const cleanBase = originalName.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');
  return `${cleanBase || 'image'}.png`;
}

/**
 * Verifies the exported Blob carries the PNG magic signature (89 50 4E 47)
 * and strict 'image/png' MIME type so zero lossy compression artifacts occur.
 */
async function ensureStrictPngBlob(blob: Blob): Promise<Blob> {
  const header = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
  const isPngSignature =
    header.length === 4 &&
    header[0] === 0x89 &&
    header[1] === 0x50 &&
    header[2] === 0x4e &&
    header[3] === 0x47;

  if (!isPngSignature) {
    throw new Error('Export validation failed: output stream is not a valid lossless PNG.');
  }

  return blob.type === LOSSLESS_PNG_MIME
    ? blob
    : new Blob([blob], { type: LOSSLESS_PNG_MIME });
}

/**
 * Stateless pseudo-random stochastic tie-breaker in [-0.25, +0.25] so
 * uniform regions never lock into repetitive regular dithering patterns.
 */
function stochasticJitter(x: number, y: number, channel: number): number {
  let h = (x * 374761393) ^ (y * 668265263) ^ (channel * 1274126177);
  h = ((h >>> 13) ^ h) * 1274126177;
  h = (h >>> 16) ^ h;
  return (((h & 0xffff) / 65535.0) - 0.5) * 0.5;
}

/**
 * Applies serpentine Floyd-Steinberg error-diffusion dithering to the continuous
 * floating-point adversarial perturbation layer (deltaR, deltaG, deltaB).
 *
 * Replaces coarse quantization plateaus with a visually invisible, fine-grained
 * high-frequency blue-noise stochastic pattern while gating diffusion across
 * sharp typography boundaries (`sharpEdgeMask`) so text edges remain razor sharp.
 *
 * Classic Floyd-Steinberg error diffusion weights:
 *         [*]   7/16
 *   3/16  5/16  1/16   (mirrored on odd serpentine scanlines)
 */
export function applyFloydSteinbergAdversarialDithering(
  sourceImageData: ImageData,
  deltaR: Float32Array,
  deltaG: Float32Array,
  deltaB: Float32Array,
  localMaxBounds: Float32Array,
  sharpEdgeMask: Float32Array
): ImageData {
  const { width, height, data: origData } = sourceImageData;

  const errR = new Float32Array(deltaR);
  const errG = new Float32Array(deltaG);
  const errB = new Float32Array(deltaB);

  const outData = new Uint8ClampedArray(origData.length);

  for (let y = 0; y < height; y++) {
    const isReverse = (y & 1) === 1;
    const xStart = isReverse ? width - 1 : 0;
    const xEnd = isReverse ? -1 : width;
    const xStep = isReverse ? -1 : 1;

    for (let x = xStart; x !== xEnd; x += xStep) {
      const i = y * width + x;
      const idx = i * 4;
      const maxEps = localMaxBounds[i];
      const isEdge = sharpEdgeMask[i] > 0.2;

      const rawDr = Math.max(-maxEps, Math.min(maxEps, errR[i]));
      const rawDg = Math.max(-maxEps, Math.min(maxEps, errG[i]));
      const rawDb = Math.max(-maxEps, Math.min(maxEps, errB[i]));

      const jR = isEdge ? 0 : stochasticJitter(x, y, 0);
      const jG = isEdge ? 0 : stochasticJitter(x, y, 1);
      const jB = isEdge ? 0 : stochasticJitter(x, y, 2);

      const targetR = Math.max(0, Math.min(255, origData[idx] + rawDr));
      const targetG = Math.max(0, Math.min(255, origData[idx + 1] + rawDg));
      const targetB = Math.max(0, Math.min(255, origData[idx + 2] + rawDb));

      const qR = Math.max(0, Math.min(255, Math.round(targetR + jR)));
      const qG = Math.max(0, Math.min(255, Math.round(targetG + jG)));
      const qB = Math.max(0, Math.min(255, Math.round(targetB + jB)));

      const clampedR = Math.max(
        Math.max(0, Math.ceil(origData[idx] - maxEps)),
        Math.min(Math.min(255, Math.floor(origData[idx] + maxEps)), qR)
      );
      const clampedG = Math.max(
        Math.max(0, Math.ceil(origData[idx + 1] - maxEps)),
        Math.min(Math.min(255, Math.floor(origData[idx + 1] + maxEps)), qG)
      );
      const clampedB = Math.max(
        Math.max(0, Math.ceil(origData[idx + 2] - maxEps)),
        Math.min(Math.min(255, Math.floor(origData[idx + 2] + maxEps)), qB)
      );

      outData[idx] = clampedR;
      outData[idx + 1] = clampedG;
      outData[idx + 2] = clampedB;
      outData[idx + 3] = 255;

      const edgeGate = isEdge ? 0.15 : 1.0;
      const qErrR = (targetR - clampedR) * edgeGate;
      const qErrG = (targetG - clampedG) * edgeGate;
      const qErrB = (targetB - clampedB) * edgeGate;

      const nextX = x + xStep;
      const prevX = x - xStep;
      const nextY = y + 1;

      if (nextX >= 0 && nextX < width) {
        const ni = y * width + nextX;
        const w = 7 / 16;
        errR[ni] += qErrR * w;
        errG[ni] += qErrG * w;
        errB[ni] += qErrB * w;
      }

      if (nextY < height) {
        if (prevX >= 0 && prevX < width) {
          const ni = nextY * width + prevX;
          const w = 3 / 16;
          errR[ni] += qErrR * w;
          errG[ni] += qErrG * w;
          errB[ni] += qErrB * w;
        }

        const bi = nextY * width + x;
        const wb = 5 / 16;
        errR[bi] += qErrR * wb;
        errG[bi] += qErrG * wb;
        errB[bi] += qErrB * wb;

        if (nextX >= 0 && nextX < width) {
          const ni = nextY * width + nextX;
          const w = 1 / 16;
          errR[ni] += qErrR * w;
          errG[ni] += qErrG * w;
          errB[ni] += qErrB * w;
        }
      }
    }
  }

  return new ImageData(outData, width, height);
}

/**
 * Primary PhotoGuard immunization entry point.
 * Synthesizes continuous adversarial perturbations, applies them via Floyd-Steinberg
 * stochastic dithering, computes real PSNR/SSIM metrics, and exports strictly as lossless 'image/png'.
 */
export async function immunizeImage(
  source: PreprocessedImage,
  options: ProtectionOptions,
  onProgress?: (progress: ProtectionProgress) => void,
  engineMode: ProcessingEngineMode = 'client'
): Promise<ProtectionResult> {
  if (engineMode === 'api') {
    return immunizeViaFastApi(source, options, onProgress);
  }

  const startTime = performance.now();
  const preset = PRESET_CONFIGS[options.strength];
  const {
    deltaR,
    deltaG,
    deltaB,
    localMaxBounds,
    sharpEdgeMask,
  } = await runClientAdversarialPerturbation(source.imageData, options, onProgress);

  onProgress?.({
    currentStep: preset.steps,
    totalSteps: preset.steps,
    phase: 'Applying Floyd-Steinberg stochastic dithering',
    percent: 90,
  });

  const protectedImageData = applyFloydSteinbergAdversarialDithering(
    source.imageData,
    deltaR,
    deltaG,
    deltaB,
    localMaxBounds,
    sharpEdgeMask
  );

  const { psnrDb, ssim, meanAbsDelta } = computeQualityMetrics(
    source.imageData.data,
    protectedImageData.data,
    source.width,
    source.height
  );

  onProgress?.({
    currentStep: preset.steps,
    totalSteps: preset.steps,
    phase: 'Encoding lossless 32-bit PNG stream',
    percent: 96,
  });

  const rawProtectedBlob = await imageDataToLosslessPngBlob(protectedImageData);
  const protectedBlob = await ensureStrictPngBlob(rawProtectedBlob);
  const protectedUrl = URL.createObjectURL(protectedBlob);

  const { diffRgbUrl, diffMagnitudeUrl } = await renderDifferenceMaps(
    source.imageData,
    protectedImageData,
    25
  );

  const processingTimeMs = Math.round(performance.now() - startTime);

  onProgress?.({
    currentStep: preset.steps,
    totalSteps: preset.steps,
    phase: 'Immunization complete',
    percent: 100,
  });

  const dynamicEpsilon255 = Math.max(2, Math.min(10, options.epsilon255));
  const dynamicEpsilon = Number((dynamicEpsilon255 / 255).toFixed(4));

  return {
    originalUrl: source.previewUrl,
    protectedUrl,
    protectedBlob,
    protectedImageData,
    diffRgbUrl,
    diffMagnitudeUrl,
    downloadFileName: formatProtectedPngFilename(source.fileName),
    metrics: {
      processingTimeMs,
      width: source.width,
      height: source.height,
      epsilon: dynamicEpsilon,
      epsilon255: dynamicEpsilon255,
      steps: preset.steps,
      eotEnabled: options.enableEot,
      psnrDb,
      ssim,
      meanAbsDelta,
      outputPngBytes: protectedBlob.size,
      formatLabel: 'PNG-32 (Lossless RGBA)',
    },
  };
}

/**
 * Drop-in FastAPI backend adapter for POST /api/protect.
 * Sends the downscaled, EXIF-stripped lossless 'image/png' blob in memory and parses
 * returned lossless PNG + quality headers from the PyTorch TAESD PGD server.
 */
async function immunizeViaFastApi(
  source: PreprocessedImage,
  options: ProtectionOptions,
  onProgress?: (progress: ProtectionProgress) => void
): Promise<ProtectionResult> {
  const preset = PRESET_CONFIGS[options.strength];
  const apiBase = import.meta.env.VITE_PHOTOGUARD_API_URL || '';

  onProgress?.({
    currentStep: 1,
    totalSteps: preset.steps,
    phase: 'Uploading lossless PNG tensor to /api/protect',
    percent: 25,
  });

  const dynamicEpsilon255 = Math.max(2, Math.min(10, options.epsilon255));
  const dynamicEpsilon = Number((dynamicEpsilon255 / 255).toFixed(4));

  const rawInputPng = await imageDataToLosslessPngBlob(source.imageData);
  const inputBlob = await ensureStrictPngBlob(rawInputPng);
  const formData = new FormData();
  formData.append('file', inputBlob, buildPngSourceFilename(source.fileName));
  formData.append('strength', options.strength);
  formData.append('epsilon', String(dynamicEpsilon255 / 255));
  formData.append('steps', String(preset.steps));
  formData.append('enable_eot', String(options.enableEot));

  const startTime = performance.now();
  const response = await fetch(`${apiBase}/api/protect`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => '');
    throw new Error(`Backend immunization failed (${response.status}): ${errText || response.statusText}`);
  }

  onProgress?.({
    currentStep: preset.steps,
    totalSteps: preset.steps,
    phase: 'Decoding lossless PNG & rendering diff map',
    percent: 88,
  });

  const rawResponseBlob = await response.blob();
  const downloadFileName = formatProtectedPngFilename(source.fileName);
  const decodedProtected = await preprocessImageBlob(
    rawResponseBlob,
    downloadFileName,
    Math.max(source.width, source.height)
  );
  const protectedBlob = await ensureStrictPngBlob(
    await imageDataToLosslessPngBlob(decodedProtected.imageData)
  );
  const protectedUrl = URL.createObjectURL(protectedBlob);

  const computed = computeQualityMetrics(
    source.imageData.data,
    decodedProtected.imageData.data,
    source.width,
    source.height
  );

  const { diffRgbUrl, diffMagnitudeUrl } = await renderDifferenceMaps(
    source.imageData,
    decodedProtected.imageData,
    25
  );

  URL.revokeObjectURL(decodedProtected.previewUrl);

  const headerTime = Number(response.headers.get('X-Process-Time-Ms'));
  const headerPsnr = Number(response.headers.get('X-PSNR'));
  const headerSsim = Number(response.headers.get('X-SSIM'));

  return {
    originalUrl: source.previewUrl,
    protectedUrl,
    protectedBlob,
    protectedImageData: decodedProtected.imageData,
    diffRgbUrl,
    diffMagnitudeUrl,
    downloadFileName,
    metrics: {
      processingTimeMs: Number.isFinite(headerTime) && headerTime > 0 ? headerTime : Math.round(performance.now() - startTime),
      width: source.width,
      height: source.height,
      epsilon: dynamicEpsilon,
      epsilon255: dynamicEpsilon255,
      steps: preset.steps,
      eotEnabled: options.enableEot,
      psnrDb: Number.isFinite(headerPsnr) && headerPsnr > 0 ? headerPsnr : computed.psnrDb,
      ssim: Number.isFinite(headerSsim) && headerSsim > 0 ? headerSsim : computed.ssim,
      meanAbsDelta: computed.meanAbsDelta,
      outputPngBytes: protectedBlob.size,
      formatLabel: 'PNG-32 (Lossless RGBA)',
    },
  };
}
