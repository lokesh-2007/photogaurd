export type ProtectionStrength = 'light' | 'balanced' | 'maximum';

export interface PresetConfig {
  id: ProtectionStrength;
  label: string;
  tag: string;
  epsilon: number; // Normalized [0, 1] L-infinity budget
  epsilon255: number; // Equivalent 8-bit integer budget (approx)
  steps: number; // PGD optimization iterations
  description: string;
}

export const PRESET_CONFIGS: Record<ProtectionStrength, PresetConfig> = {
  light: {
    id: 'light',
    label: 'Light',
    tag: 'Fast',
    epsilon: 0.03,
    epsilon255: 8,
    steps: 15,
    description: 'Low-latency L∞ perturbation preserving near-identical pixel fidelity.',
  },
  balanced: {
    id: 'balanced',
    label: 'Balanced',
    tag: 'Recommended',
    epsilon: 0.05,
    epsilon255: 13,
    steps: 20,
    description: 'Optimal trade-off between diffusion encoder disruption and visual invisibility.',
  },
  maximum: {
    id: 'maximum',
    label: 'Maximum',
    tag: 'Strongest',
    epsilon: 0.08,
    epsilon255: 20,
    steps: 25,
    description: 'Aggressive L∞ budget hardened against aggressive generative inpainting.',
  },
};

export interface PreprocessedImage {
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
}

export interface ProtectionOptions {
  strength: ProtectionStrength;
  enableEot: boolean;
  highDpiMode: boolean;
}

export interface ProtectionProgress {
  currentStep: number;
  totalSteps: number;
  phase: string;
  percent: number;
}

export interface ProtectionMetrics {
  processingTimeMs: number;
  width: number;
  height: number;
  epsilon: number;
  epsilon255: number;
  steps: number;
  eotEnabled: boolean;
  psnrDb: number;
  ssim: number;
  meanAbsDelta: number;
  outputPngBytes: number;
  formatLabel: string;
}

export interface ProtectionResult {
  originalUrl: string;
  protectedUrl: string;
  protectedBlob: Blob;
  protectedImageData: ImageData;
  diffRgbUrl: string;
  diffMagnitudeUrl: string;
  downloadFileName: string;
  metrics: ProtectionMetrics;
}

export type ViewportMode = 'split' | 'inspector' | 'side-by-side';
export type InspectorChannelMode = 'rgb' | 'magnitude';
