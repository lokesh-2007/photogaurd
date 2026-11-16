import { useCallback, useEffect, useRef, useState } from 'react';
import { BackendSpecModal } from './components/BackendSpecModal';
import { ControlPanel } from './components/ControlPanel';
import { Header } from './components/Header';
import { UploadZone } from './components/UploadZone';
import { VerificationStage } from './components/VerificationStage';
import { formatProtectedPngFilename, immunizeImage } from './services/immunizeService';
import {
  PreprocessedImage,
  ProtectionOptions,
  ProtectionProgress,
  ProtectionResult,
} from './types/photoguard';
import {
  DEFAULT_MAX_LONG_EDGE,
  HIGH_DPI_MAX_LONG_EDGE,
  preprocessImageBlob,
  preprocessImageFromUrl,
} from './utils/imageProcessing';

const ACCEPTED_MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

interface RawSourceRef {
  blob?: Blob;
  url?: string;
  fileName: string;
}

export default function App() {
  const [sourceImage, setSourceImage] = useState<PreprocessedImage | null>(null);
  const [result, setResult] = useState<ProtectionResult | null>(null);
  const [options, setOptions] = useState<ProtectionOptions>({
    strength: 'balanced',
    enableEot: true,
    highDpiMode: true,
  });

  const [isLoadingSource, setIsLoadingSource] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState<ProtectionProgress | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isBackendModalOpen, setIsBackendModalOpen] = useState(false);

  const rawSourceRef = useRef<RawSourceRef | null>(null);
  const activeSourceUrlRef = useRef<string | null>(null);
  const activeResultUrlsRef = useRef<string[]>([]);

  const revokeResultUrls = useCallback(() => {
    for (const url of activeResultUrlsRef.current) {
      URL.revokeObjectURL(url);
    }
    activeResultUrlsRef.current = [];
  }, []);

  const revokeSourceUrl = useCallback(() => {
    if (activeSourceUrlRef.current) {
      URL.revokeObjectURL(activeSourceUrlRef.current);
      activeSourceUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      revokeResultUrls();
      revokeSourceUrl();
    };
  }, [revokeResultUrls, revokeSourceUrl]);

  const applyNewSource = useCallback(
    (nextSource: PreprocessedImage) => {
      revokeResultUrls();
      revokeSourceUrl();
      activeSourceUrlRef.current = nextSource.previewUrl;
      setResult(null);
      setSourceImage(nextSource);
      setErrorMessage(null);
    },
    [revokeResultUrls, revokeSourceUrl]
  );

  const getMaxEdge = (highDpi: boolean) =>
    highDpi ? HIGH_DPI_MAX_LONG_EDGE : DEFAULT_MAX_LONG_EDGE;

  const handleSelectFile = useCallback(
    async (file: File) => {
      if (file.type && !ACCEPTED_MIME_TYPES.has(file.type)) {
        setErrorMessage('Unsupported format. Please upload a PNG, JPG, or WebP image.');
        return;
      }

      setIsLoadingSource(true);
      setErrorMessage(null);
      try {
        rawSourceRef.current = { blob: file, fileName: file.name };
        const preprocessed = await preprocessImageBlob(
          file,
          file.name,
          getMaxEdge(options.highDpiMode)
        );
        applyNewSource(preprocessed);
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : 'Unable to process selected image.');
      } finally {
        setIsLoadingSource(false);
      }
    },
    [applyNewSource, options.highDpiMode]
  );

  const handleSelectSample = useCallback(
    async (url: string, fileName: string) => {
      setIsLoadingSource(true);
      setErrorMessage(null);
      try {
        rawSourceRef.current = { url, fileName };
        const preprocessed = await preprocessImageFromUrl(
          url,
          fileName,
          getMaxEdge(options.highDpiMode)
        );
        applyNewSource(preprocessed);
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : 'Unable to load sample image.');
      } finally {
        setIsLoadingSource(false);
      }
    },
    [applyNewSource, options.highDpiMode]
  );

  const handleChangeOptions = useCallback(
    async (nextOptions: ProtectionOptions) => {
      const dpiChanged = nextOptions.highDpiMode !== options.highDpiMode;
      setOptions(nextOptions);

      // Re-decode in-memory raw source if High-DPI toggle changed while an image is loaded
      if (dpiChanged && rawSourceRef.current) {
        const raw = rawSourceRef.current;
        setIsLoadingSource(true);
        try {
          const maxEdge = getMaxEdge(nextOptions.highDpiMode);
          const redecoded = raw.blob
            ? await preprocessImageBlob(raw.blob, raw.fileName, maxEdge)
            : raw.url
            ? await preprocessImageFromUrl(raw.url, raw.fileName, maxEdge)
            : null;
          if (redecoded) {
            applyNewSource(redecoded);
          }
        } catch (err) {
          setErrorMessage(err instanceof Error ? err.message : 'Failed to adjust resolution.');
        } finally {
          setIsLoadingSource(false);
        }
      }
    },
    [options.highDpiMode, applyNewSource]
  );

  const handleClearImage = useCallback(() => {
    rawSourceRef.current = null;
    revokeResultUrls();
    revokeSourceUrl();
    setResult(null);
    setSourceImage(null);
    setProgress(null);
    setErrorMessage(null);
  }, [revokeResultUrls, revokeSourceUrl]);

  const handleImmunize = useCallback(async () => {
    if (!sourceImage || isProcessing) return;

    setIsProcessing(true);
    setErrorMessage(null);
    setProgress({
      currentStep: 0,
      totalSteps: 1,
      phase: 'Initializing adversarial tensor pipeline',
      percent: 2,
    });

    try {
      const nextResult = await immunizeImage(sourceImage, options, (p) => {
        setProgress(p);
      });

      revokeResultUrls();
      activeResultUrlsRef.current = [
        nextResult.protectedUrl,
        nextResult.diffRgbUrl,
        nextResult.diffMagnitudeUrl,
      ];
      setResult(nextResult);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Immunization failed.');
    } finally {
      setIsProcessing(false);
    }
  }, [sourceImage, options, isProcessing, revokeResultUrls]);

  const handleDownload = useCallback(() => {
    if (!result) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = result.protectedImageData.width;
    exportCanvas.height = result.protectedImageData.height;

    const ctx = exportCanvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    ctx.imageSmoothingEnabled = false;
    ctx.globalCompositeOperation = 'copy';
    ctx.putImageData(result.protectedImageData, 0, 0);

    // Strictly export lossless 32-bit PNG (never image/jpeg)
    const dataUrl = exportCanvas.toDataURL('image/png');
    const originalFileName = sourceImage?.fileName ?? result.downloadFileName;

    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = formatProtectedPngFilename(originalFileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [result, sourceImage]);

  return (
    <div className="flex min-h-screen flex-col bg-[#090B10] text-zinc-100">
      <Header onOpenBackendSpec={() => setIsBackendModalOpen(true)} />

      <main className="mx-auto w-full max-w-[1440px] flex-1 px-5 py-6 sm:px-8">
        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[370px_1fr]">
          {/* Left Control & Upload Column */}
          <aside className="rounded-xl border border-zinc-800/90 bg-[#10131B] p-5 space-y-6">
            <UploadZone
              sourceImage={sourceImage}
              isLoadingSource={isLoadingSource}
              onSelectFile={handleSelectFile}
              onSelectSample={handleSelectSample}
              onClearImage={handleClearImage}
              errorMessage={errorMessage}
            />

            <ControlPanel
              options={options}
              onChangeOptions={handleChangeOptions}
              hasSourceImage={Boolean(sourceImage)}
              isProcessing={isProcessing}
              progress={progress}
              onImmunize={handleImmunize}
            />
          </aside>

          {/* Main Interactive Verification Viewport */}
          <section aria-label="Image verification and inspection stage" className="min-w-0">
            <VerificationStage
              sourceImage={sourceImage}
              result={result}
              isProcessing={isProcessing}
              onDownload={handleDownload}
            />
          </section>
        </div>
      </main>

      <BackendSpecModal
        isOpen={isBackendModalOpen}
        onClose={() => setIsBackendModalOpen(false)}
      />
    </div>
  );
}
