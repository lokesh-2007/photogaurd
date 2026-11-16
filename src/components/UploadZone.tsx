import React, { useRef, useState } from 'react';
import { ImagePlus, Upload, X, Check } from 'lucide-react';
import { PreprocessedImage } from '../types/photoguard';
import samplePortraitUrl from '../assets/images/sample_portrait_studio_1794800985498.jpg';
import sampleUrbanUrl from '../assets/images/sample_urban_street_1794800986263.jpg';

interface SampleAsset {
  id: string;
  label: string;
  fileName: string;
  url: string;
}

const SAMPLE_ASSETS: SampleAsset[] = [
  {
    id: 'portrait',
    label: 'Studio Portrait',
    fileName: 'studio_portrait.jpg',
    url: samplePortraitUrl,
  },
  {
    id: 'urban',
    label: 'Urban Architecture',
    fileName: 'urban_dusk.jpg',
    url: sampleUrbanUrl,
  },
];

interface UploadZoneProps {
  sourceImage: PreprocessedImage | null;
  isLoadingSource: boolean;
  onSelectFile: (file: File) => void;
  onSelectSample: (url: string, fileName: string) => void;
  onClearImage: () => void;
  errorMessage: string | null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function UploadZone({
  sourceImage,
  isLoadingSource,
  onSelectFile,
  onSelectSample,
  onClearImage,
  errorMessage,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileList = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    onSelectFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileList(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  return (
    <section aria-labelledby="upload-section-heading" className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 id="upload-section-heading" className="text-sm font-semibold text-zinc-100">
          01. Source Photograph
        </h2>
        {sourceImage && (
          <button
            type="button"
            onClick={onClearImage}
            className="inline-flex items-center gap-1 text-xs font-medium text-zinc-400 transition-colors hover:text-zinc-200 focus-visible:outline-2 focus-visible:outline-cyan-500 cursor-pointer"
          >
            <X className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Clear</span>
          </button>
        )}
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload photo by dropping a file or pressing Enter to browse"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`group relative flex flex-col items-center justify-center rounded-lg border border-dashed px-4 py-5 text-center transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-500 ${
          isDragging
            ? 'border-emerald-400 bg-emerald-500/10'
            : 'border-zinc-700/90 bg-[#0D1017] hover:border-zinc-500 hover:bg-[#121620]'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="sr-only"
          onChange={(e) => {
            handleFileList(e.target.files);
            e.target.value = '';
          }}
        />

        <div className="mb-2.5 flex h-9 w-9 items-center justify-center rounded-md border border-zinc-800 bg-zinc-900 text-zinc-300 transition-colors group-hover:border-emerald-500/40 group-hover:text-emerald-400">
          {isLoadingSource ? (
            <Upload className="h-4 w-4 animate-pulse text-cyan-400" aria-hidden="true" />
          ) : (
            <ImagePlus className="h-4 w-4" aria-hidden="true" />
          )}
        </div>

        <p className="text-sm font-medium text-zinc-200">
          {isLoadingSource ? 'Decoding & stripping metadata...' : 'Drop photo here or browse'}
        </p>
        <p className="mt-1 text-xs text-zinc-400">
          PNG, JPG, WebP · Lossless 32-bit PNG output
        </p>
      </div>

      {errorMessage && (
        <p role="alert" className="rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
          {errorMessage}
        </p>
      )}

      {sourceImage && (
        <div className="rounded-lg border border-zinc-800/90 bg-[#0D1017] px-3.5 py-3 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate font-medium text-zinc-200" title={sourceImage.fileName}>
              {sourceImage.fileName}
            </span>
            <span className="inline-flex items-center gap-1 font-mono text-emerald-400 shrink-0">
              <Check className="h-3.5 w-3.5" aria-hidden="true" />
              EXIF Stripped
            </span>
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[11px] text-zinc-400 tabular-nums">
            <span>
              {sourceImage.wasDownscaled
                ? `${sourceImage.originalWidth}×${sourceImage.originalHeight} → ${sourceImage.width}×${sourceImage.height} px`
                : `${sourceImage.width}×${sourceImage.height} px`}
            </span>
            <span aria-hidden="true">·</span>
            <span>{formatBytes(sourceImage.sourceByteSize)}</span>
          </div>
        </div>
      )}

      <div className="space-y-2 pt-1">
        <span className="block text-xs font-medium text-zinc-400">
          Or load a test sample
        </span>
        <div className="grid grid-cols-2 gap-2.5">
          {SAMPLE_ASSETS.map((sample) => {
            const isActive = sourceImage?.fileName === sample.fileName;
            return (
              <button
                key={sample.id}
                type="button"
                disabled={isLoadingSource}
                onClick={() => onSelectSample(sample.url, sample.fileName)}
                className={`group flex items-center gap-2.5 rounded-md border p-1.5 text-left transition-colors focus-visible:outline-2 focus-visible:outline-cyan-500 cursor-pointer disabled:opacity-50 ${
                  isActive
                    ? 'border-emerald-500/60 bg-emerald-500/10'
                    : 'border-zinc-800 bg-[#0D1017] hover:border-zinc-700 hover:bg-zinc-900/80'
                }`}
              >
                <div className="h-9 w-11 shrink-0 overflow-hidden rounded bg-zinc-900">
                  <img
                    src={sample.url}
                    alt={sample.label}
                    referrerPolicy="no-referrer"
                    className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105"
                  />
                </div>
                <span className="truncate text-xs font-medium text-zinc-200">
                  {sample.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
