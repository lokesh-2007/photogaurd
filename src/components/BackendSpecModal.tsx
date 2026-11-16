import { useEffect } from 'react';
import { Check, Copy, X } from 'lucide-react';
import { useState } from 'react';

interface BackendSpecModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const FASTAPI_SNIPPET = `# 1. Start the FastAPI + PyTorch TAESD (float16) backend
cd backend && pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. Swap client processor to POST /api/protect in src/services/immunizeService.ts
immunizeImage(sourceImage, options, onProgress, 'api');`;

export function BackendSpecModal({ isOpen, onClose }: BackendSpecModalProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(FASTAPI_SNIPPET);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard fallback ignored cleanly
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="backend-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl rounded-xl border border-zinc-800 bg-[#10131B] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
          <h2 id="backend-modal-title" className="text-base font-semibold text-zinc-100">
            Architecture & FastAPI PyTorch Adapter
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close modal"
            className="rounded-md p-1 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100 focus-visible:outline-2 focus-visible:outline-cyan-500 cursor-pointer"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <div className="space-y-4 px-5 py-5 text-sm leading-relaxed text-zinc-300">
          <p>
            This live preview executes an in-browser HTML5 Canvas adversarial engine—combining JND typography edge protection, Floyd-Steinberg stochastic dithering, L∞ clamping, optional EOT frequency hardening, and real PSNR/SSIM evaluation—with zero disk or server retention.
          </p>

          <div className="grid gap-3 sm:grid-cols-2 text-xs">
            <div className="rounded-lg border border-zinc-800 bg-[#090B10] p-3.5">
              <div className="font-semibold text-emerald-400">Zero-Retention &amp; Lossless PNG-32</div>
              <p className="mt-1 text-zinc-400">
                Uploaded photos are decoded into raw RGBA pixels in memory, stripping EXIF/GPS metadata automatically, and exported strictly as <code className="text-zinc-200">image/png</code>.
              </p>
            </div>

            <div className="rounded-lg border border-zinc-800 bg-[#090B10] p-3.5">
              <div className="font-semibold text-cyan-400">EOT + Cross-Model Adam-PGD</div>
              <p className="mt-1 text-zinc-400">
                <code className="text-zinc-200">/backend/main.py</code> runs a 30-step Adam-PGD loop (<code className="text-zinc-200">ε ≤ 6/255</code>) with differentiable JPEG/scale/blur EOT across <code className="text-zinc-200">taesd</code>, <code className="text-zinc-200">taesdxl</code>, and optional UNet down-blocks.
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-400">
                Swapping to GPU/CPU FastAPI Backend
              </span>
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 rounded border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-xs text-zinc-200 transition-colors hover:bg-zinc-800 cursor-pointer"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" />
                    <span>Copy Commands</span>
                  </>
                )}
              </button>
            </div>
            <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-[#090B10] p-3.5 font-mono text-xs text-zinc-200">
              {FASTAPI_SNIPPET}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
