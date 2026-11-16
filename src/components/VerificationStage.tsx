import React, { useCallback, useRef, useState } from 'react';
import {
  Columns2,
  Download,
  Eye,
  MoveHorizontal,
  SlidersHorizontal,
  UploadCloud,
} from 'lucide-react';
import {
  InspectorChannelMode,
  PreprocessedImage,
  ProtectionResult,
  ViewportMode,
} from '../types/photoguard';

interface VerificationStageProps {
  sourceImage: PreprocessedImage | null;
  result: ProtectionResult | null;
  isProcessing: boolean;
  onDownload: () => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function VerificationStage({
  sourceImage,
  result,
  isProcessing,
  onDownload,
}: VerificationStageProps) {
  const [viewMode, setViewMode] = useState<ViewportMode>('split');
  const [inspectorChannel, setInspectorChannel] = useState<InspectorChannelMode>('rgb');
  const [splitPercent, setSplitPercent] = useState<number>(50);
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false);

  const stageRef = useRef<HTMLDivElement>(null);

  const updateSplitFromClientX = useCallback((clientX: number) => {
    const el = stageRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    const clamped = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSplitPercent(Math.round(clamped * 10) / 10);
  }, []);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (viewMode !== 'split' || !result) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setIsDraggingSplit(true);
    updateSplitFromClientX(e.clientX);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingSplit) return;
    updateSplitFromClientX(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingSplit) return;
    e.currentTarget.releasePointerCapture(e.pointerId);
    setIsDraggingSplit(false);
  };

  const handleSliderKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (viewMode !== 'split' || !result) return;
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setSplitPercent((p) => Math.max(0, p - 2));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setSplitPercent((p) => Math.min(100, p + 2));
    }
  };

  return (
    <div className="flex h-full flex-col rounded-xl border border-zinc-800/90 bg-[#10131B]">
      {/* Top Verification Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/90 px-4 py-3 sm:px-5">
        <div className="flex items-center gap-1 rounded-lg border border-zinc-800 bg-[#090B10] p-1">
          <button
            type="button"
            disabled={!result}
            onClick={() => setViewMode('split')}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${
              viewMode === 'split'
                ? 'bg-zinc-800 text-zinc-50'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <SlidersHorizontal className="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" />
            <span>Split Compare</span>
          </button>

          <button
            type="button"
            disabled={!result}
            onClick={() => setViewMode('inspector')}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${
              viewMode === 'inspector'
                ? 'bg-zinc-800 text-zinc-50'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Eye className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
            <span>Noise Inspector (×25)</span>
          </button>

          <button
            type="button"
            disabled={!result}
            onClick={() => setViewMode('side-by-side')}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${
              viewMode === 'side-by-side'
                ? 'bg-zinc-800 text-zinc-50'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Columns2 className="h-3.5 w-3.5 text-zinc-300" aria-hidden="true" />
            <span>Side-by-Side</span>
          </button>
        </div>

        <div className="flex items-center gap-3">
          {viewMode === 'inspector' && result && (
            <div className="flex items-center gap-1 rounded-md border border-zinc-800 bg-[#090B10] p-0.5">
              <button
                type="button"
                onClick={() => setInspectorChannel('rgb')}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  inspectorChannel === 'rgb'
                    ? 'bg-zinc-800 text-cyan-300'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                RGB ×25
              </button>
              <button
                type="button"
                onClick={() => setInspectorChannel('magnitude')}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  inspectorChannel === 'magnitude'
                    ? 'bg-zinc-800 text-emerald-300'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Magnitude Map
              </button>
            </div>
          )}

          <button
            type="button"
            disabled={!result || isProcessing}
            onClick={onDownload}
            className="inline-flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/15 px-3.5 py-1.5 text-xs font-semibold text-emerald-300 transition-colors hover:bg-emerald-500/25 focus-visible:outline-2 focus-visible:outline-emerald-400 whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            <span>
              Download Lossless PNG
              {result ? ` (${formatFileSize(result.metrics.outputPngBytes)})` : ''}
            </span>
          </button>
        </div>
      </div>

      {/* Interactive Image Canvas Stage */}
      <div className="inspection-grid-bg relative flex min-h-[420px] flex-1 items-center justify-center overflow-hidden p-4 sm:min-h-[520px] sm:p-6">
        {!sourceImage && (
          <div className="max-w-md space-y-3 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg border border-zinc-800 bg-[#10131B] text-zinc-400">
              <UploadCloud className="h-5 w-5" aria-hidden="true" />
            </div>
            <h3 className="text-base font-semibold text-zinc-200">
              No photograph loaded
            </h3>
            <p className="text-sm leading-relaxed text-zinc-400">
              Upload a PNG, JPG, or WebP photo on the left—or pick one of the studio samples—to inspect and immunize it in memory.
            </p>
          </div>
        )}

        {sourceImage && !result && (
          <div className="relative flex max-h-[540px] max-w-full flex-col items-center">
            <img
              src={sourceImage.previewUrl}
              alt="Source preview prior to immunization"
              referrerPolicy="no-referrer"
              className="max-h-[500px] w-auto rounded-md border border-zinc-800 object-contain shadow-2xl"
            />
            <div className="mt-3 flex items-center gap-3 font-mono text-xs text-zinc-400 tabular-nums">
              <span>Unprotected Preview</span>
              <span aria-hidden="true">·</span>
              <span>
                {sourceImage.width} × {sourceImage.height} px
              </span>
              <span aria-hidden="true">·</span>
              <span className="text-emerald-400">Ready to immunize</span>
            </div>
          </div>
        )}

        {sourceImage && result && viewMode === 'split' && (
          <div className="flex flex-col items-center w-full">
            <div
              ref={stageRef}
              role="slider"
              tabIndex={0}
              aria-label="Before and after split comparison slider"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(splitPercent)}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onKeyDown={handleSliderKeyDown}
              className="relative inline-block max-h-[520px] max-w-full select-none overflow-hidden rounded-md border border-zinc-700/80 shadow-2xl cursor-ew-resize focus-visible:outline-2 focus-visible:outline-cyan-400 touch-none"
            >
              {/* Base Layer: Protected Image */}
              <img
                src={result.protectedUrl}
                alt="Protected immunized photograph"
                draggable={false}
                referrerPolicy="no-referrer"
                className="block max-h-[520px] w-auto max-w-full object-contain pointer-events-none"
              />

              {/* Clipped Overlay Layer: Original Image */}
              <div
                className="absolute inset-0 overflow-hidden pointer-events-none"
                style={{ clipPath: `inset(0 ${100 - splitPercent}% 0 0)` }}
              >
                <img
                  src={result.originalUrl}
                  alt="Original unmodified photograph"
                  draggable={false}
                  referrerPolicy="no-referrer"
                  className="block h-full w-full object-contain"
                />
              </div>

              {/* Divider Line & Handle */}
              <div
                className="absolute inset-y-0 w-0.5 bg-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.7)] pointer-events-none"
                style={{ left: `${splitPercent}%` }}
              >
                <div className="absolute top-1/2 left-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-emerald-300 bg-[#090B10]/95 text-emerald-300 shadow-lg">
                  <MoveHorizontal className="h-4 w-4" aria-hidden="true" />
                </div>
              </div>

              {/* Corner Labels */}
              <div className="pointer-events-none absolute top-3 left-3 rounded bg-[#090B10]/85 px-2.5 py-1 font-mono text-[11px] font-medium text-zinc-200 border border-zinc-700/70">
                Original
              </div>
              <div className="pointer-events-none absolute top-3 right-3 rounded bg-[#090B10]/85 px-2.5 py-1 font-mono text-[11px] font-medium text-emerald-300 border border-emerald-500/40">
                Protected (PNG)
              </div>
            </div>

            <div className="mt-3 flex items-center gap-3 font-mono text-xs text-zinc-400 tabular-nums">
              <span>Drag divider or use Arrow keys</span>
              <span aria-hidden="true">·</span>
              <span className="text-zinc-200">Split: {Math.round(splitPercent)}%</span>
            </div>
          </div>
        )}

        {sourceImage && result && viewMode === 'inspector' && (
          <div className="flex flex-col items-center w-full">
            <div className="relative inline-block max-h-[520px] max-w-full overflow-hidden rounded-md border border-cyan-500/40 bg-black shadow-2xl">
              <img
                src={inspectorChannel === 'rgb' ? result.diffRgbUrl : result.diffMagnitudeUrl}
                alt="Amplified 25x adversarial difference map"
                referrerPolicy="no-referrer"
                className="block max-h-[520px] w-auto max-w-full object-contain"
              />
              <div className="pointer-events-none absolute top-3 left-3 rounded border border-cyan-500/40 bg-[#090B10]/90 px-2.5 py-1 font-mono text-[11px] font-medium text-cyan-300">
                |Protected − Original| × 25
              </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-center gap-4 font-mono text-xs text-zinc-400 tabular-nums">
              <span>Perturbation Energy Concentrated on Semantic Contours</span>
              <span aria-hidden="true">·</span>
              <span>Mean |Δ| = {result.metrics.meanAbsDelta} / 255</span>
            </div>
          </div>
        )}

        {sourceImage && result && viewMode === 'side-by-side' && (
          <div className="grid w-full max-w-5xl grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col items-center space-y-2">
              <div className="overflow-hidden rounded-md border border-zinc-800 bg-black">
                <img
                  src={result.originalUrl}
                  alt="Original unmodified image"
                  referrerPolicy="no-referrer"
                  className="max-h-[440px] w-auto object-contain"
                />
              </div>
              <span className="font-mono text-xs text-zinc-400">Original (EXIF Stripped)</span>
            </div>

            <div className="flex flex-col items-center space-y-2">
              <div className="overflow-hidden rounded-md border border-emerald-500/40 bg-black">
                <img
                  src={result.protectedUrl}
                  alt="Protected immunized image"
                  referrerPolicy="no-referrer"
                  className="max-h-[440px] w-auto object-contain"
                />
              </div>
              <span className="font-mono text-xs text-emerald-400">
                Protected Lossless PNG
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Metrics Bar */}
      <div className="grid grid-cols-2 divide-y divide-zinc-800/80 border-t border-zinc-800/90 bg-[#0B0E14] sm:grid-cols-4 sm:divide-x sm:divide-y-0">
        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-zinc-400">Processing Time</div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-zinc-100 tabular-nums">
            {result ? `${result.metrics.processingTimeMs} ms` : '—'}
          </div>
        </div>

        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-zinc-400">Output Resolution · Stream</div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-zinc-100 tabular-nums">
            {result
              ? `${result.metrics.width} × ${result.metrics.height} · PNG-32`
              : sourceImage
              ? `${sourceImage.width} × ${sourceImage.height}`
              : '—'}
          </div>
        </div>

        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-zinc-400">Perturbation Budget (L∞)</div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-cyan-400 tabular-nums">
            {result
              ? `ε = ${result.metrics.epsilon.toFixed(2)} (±${result.metrics.epsilon255}/255)`
              : '—'}
          </div>
        </div>

        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-zinc-400">Visual Fidelity (PSNR / SSIM)</div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-emerald-400 tabular-nums">
            {result
              ? `${result.metrics.psnrDb.toFixed(2)} dB · ${result.metrics.ssim.toFixed(4)}`
              : '—'}
          </div>
        </div>
      </div>
    </div>
  );
}
