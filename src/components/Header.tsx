import { ShieldCheck, Terminal } from 'lucide-react';

interface HeaderProps {
  onOpenBackendSpec: () => void;
}

export function Header({ onOpenBackendSpec }: HeaderProps) {
  return (
    <header className="border-b border-zinc-800/90 bg-[#0B0E14]/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-5 py-3.5 sm:px-8">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-md border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
              <ShieldCheck className="h-4.5 w-4.5" aria-hidden="true" />
            </div>
            <h1 className="text-lg font-semibold tracking-tight text-zinc-50 whitespace-nowrap">
              PhotoGuard Lite
            </h1>
          </div>
          <p className="text-sm text-zinc-400">
            Imperceptible adversarial pixel immunization against unauthorized AI editing and deepfakes
          </p>
        </div>

        <div className="flex items-center gap-4 shrink-0">
          <div
            className="flex items-center gap-2 rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-1.5 text-xs font-medium text-emerald-400 whitespace-nowrap"
            title="Images are processed strictly in browser memory and never written to disk or sent to external storage."
          >
            <ShieldCheck className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span>Zero-Retention / In-Memory Only</span>
          </div>

          <button
            type="button"
            onClick={onOpenBackendSpec}
            className="inline-flex items-center gap-2 rounded-md border border-zinc-700/80 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:border-zinc-600 hover:bg-zinc-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-500 whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Terminal className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
            <span>FastAPI Spec</span>
          </button>
        </div>
      </div>
    </header>
  );
}
