import { Loader2, Shield } from 'lucide-react';
import {
  PRESET_CONFIGS,
  ProtectionOptions,
  ProtectionProgress,
  ProtectionStrength,
} from '../types/photoguard';

interface ControlPanelProps {
  options: ProtectionOptions;
  onChangeOptions: (next: ProtectionOptions) => void;
  hasSourceImage: boolean;
  isProcessing: boolean;
  progress: ProtectionProgress | null;
  onImmunize: () => void;
}

const STRENGTH_ORDER: ProtectionStrength[] = ['light', 'balanced', 'maximum'];

export function ControlPanel({
  options,
  onChangeOptions,
  hasSourceImage,
  isProcessing,
  progress,
  onImmunize,
}: ControlPanelProps) {
  return (
    <div className="space-y-6">
      <section aria-labelledby="strength-section-heading" className="space-y-3 border-t border-zinc-800/80 pt-5">
        <div className="flex items-baseline justify-between">
          <h2 id="strength-section-heading" className="text-sm font-semibold text-zinc-100">
            02. Protection Strength
          </h2>
          <span className="font-mono text-xs text-zinc-400 tabular-nums">
            L∞ Budget
          </span>
        </div>

        <div role="radiogroup" aria-labelledby="strength-section-heading" className="space-y-2">
          {STRENGTH_ORDER.map((key) => {
            const preset = PRESET_CONFIGS[key];
            const selected = options.strength === key;

            return (
              <button
                key={key}
                type="button"
                role="radio"
                aria-checked={selected}
                disabled={isProcessing}
                onClick={() =>
                  onChangeOptions({
                    ...options,
                    strength: key,
                    epsilon255: preset.epsilon255,
                  })
                }
                className={`w-full rounded-lg border px-3.5 py-2.5 text-left transition-colors focus-visible:outline-2 focus-visible:outline-cyan-500 cursor-pointer disabled:opacity-60 ${
                  selected
                    ? 'border-emerald-500/70 bg-emerald-500/10'
                    : 'border-zinc-800 bg-[#0D1017] hover:border-zinc-700 hover:bg-zinc-900/70'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-3.5 w-3.5 rounded-full border flex items-center justify-center ${
                        selected ? 'border-emerald-400' : 'border-zinc-600'
                      }`}
                    >
                      {selected && <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />}
                    </span>
                    <span className="text-sm font-medium text-zinc-100">
                      {preset.label} ({preset.tag})
                    </span>
                  </div>
                  <span className="font-mono text-xs text-cyan-400 tabular-nums shrink-0">
                    {preset.epsilon255}/255 · {preset.steps} steps
                  </span>
                </div>
                <p className="mt-1 pl-5.5 text-xs leading-relaxed text-zinc-400">
                  {preset.description}
                </p>
              </button>
            );
          })}
        </div>

        {/* Dynamic Noise Intensity Slider (2/255 to 10/255) */}
        <div className="rounded-lg border border-zinc-800 bg-[#0D1017] px-3.5 py-3 space-y-2.5">
          <div className="flex items-center justify-between gap-2">
            <label htmlFor="noise-intensity-slider" className="text-xs font-medium text-zinc-200">
              Noise Intensity (L∞ Budget)
            </label>
            <span className="font-mono text-xs font-semibold text-emerald-400 tabular-nums">
              {Number.isInteger(options.epsilon255)
                ? options.epsilon255
                : options.epsilon255.toFixed(1)}
              /255 · ε={(options.epsilon255 / 255).toFixed(4)}
            </span>
          </div>

          <input
            id="noise-intensity-slider"
            type="range"
            min={2}
            max={10}
            step={0.5}
            value={options.epsilon255}
            disabled={isProcessing}
            onChange={(e) =>
              onChangeOptions({
                ...options,
                epsilon255: parseFloat(e.target.value),
              })
            }
            className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-zinc-800 accent-emerald-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
          />

          <div className="flex items-center justify-between font-mono text-[11px] text-zinc-500 tabular-nums">
            <span>2/255 · Subtle</span>
            <span>6/255 · Balanced</span>
            <span>10/255 · Max</span>
          </div>
        </div>
      </section>

      <section aria-labelledby="robustness-section-heading" className="space-y-3 border-t border-zinc-800/80 pt-5">
        <h2 id="robustness-section-heading" className="text-sm font-semibold text-zinc-100">
          03. Defense &amp; Fidelity Options
        </h2>

        <label
          htmlFor="eot-toggle"
          className={`flex items-start gap-3 rounded-lg border px-3.5 py-3 transition-colors cursor-pointer ${
            options.enableEot
              ? 'border-cyan-500/50 bg-cyan-500/5'
              : 'border-zinc-800 bg-[#0D1017] hover:border-zinc-700'
          }`}
        >
          <input
            id="eot-toggle"
            type="checkbox"
            checked={options.enableEot}
            disabled={isProcessing}
            onChange={(e) => onChangeOptions({ ...options, enableEot: e.target.checked })}
            className="mt-0.5 h-4 w-4 rounded border-zinc-600 bg-zinc-900 text-emerald-500 focus:ring-cyan-500 focus:ring-offset-0 accent-emerald-500 cursor-pointer"
          />
          <div className="space-y-1">
            <span className="block text-sm font-medium text-zinc-100">
              Social Media Compression Defense (EOT)
            </span>
            <span className="block text-xs leading-relaxed text-zinc-400">
              Applies Expectation Over Transformation frequency hardening without halo blur so perturbations survive JPEG re-encoding.
            </span>
          </div>
        </label>

        <label
          htmlFor="high-dpi-toggle"
          className={`flex items-start gap-3 rounded-lg border px-3.5 py-3 transition-colors cursor-pointer ${
            options.highDpiMode
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-zinc-800 bg-[#0D1017] hover:border-zinc-700'
          }`}
        >
          <input
            id="high-dpi-toggle"
            type="checkbox"
            checked={options.highDpiMode}
            disabled={isProcessing}
            onChange={(e) => onChangeOptions({ ...options, highDpiMode: e.target.checked })}
            className="mt-0.5 h-4 w-4 rounded border-zinc-600 bg-zinc-900 text-emerald-500 focus:ring-cyan-500 focus:ring-offset-0 accent-emerald-500 cursor-pointer"
          />
          <div className="space-y-1">
            <span className="block text-sm font-medium text-zinc-100">
              High-DPI Document Clarity (up to 2048px)
            </span>
            <span className="block text-xs leading-relaxed text-zinc-400">
              Preserves razor-sharp small typography and fine seals on certificates and scanned documents.
            </span>
          </div>
        </label>
      </section>

      <div className="border-t border-zinc-800/80 pt-5 space-y-3">
        <button
          type="button"
          disabled={!hasSourceImage || isProcessing}
          onClick={onImmunize}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-500 px-4 py-3 text-sm font-semibold text-zinc-950 transition-colors hover:bg-emerald-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-400 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500 cursor-pointer"
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              <span>Immunizing Image...</span>
            </>
          ) : (
            <>
              <Shield className="h-4 w-4" aria-hidden="true" />
              <span>Immunize Image</span>
            </>
          )}
        </button>

        {isProcessing && progress && (
          <div className="space-y-1.5" aria-live="polite">
            <div className="flex items-center justify-between text-xs">
              <span className="truncate text-zinc-300">{progress.phase}</span>
              <span className="font-mono text-emerald-400 tabular-nums shrink-0">
                {progress.percent}%
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-[width] duration-150 ease-out"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
