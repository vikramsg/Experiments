import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { FileDown, FileText, Sparkles, Upload } from 'lucide-react';
import { PDFDocument } from 'pdf-lib';

type PdfInfo = {
  name: string;
  sizeBytes: number;
  sizeLabel: string;
  pages: number | null;
  version: string;
  title?: string;
  author?: string;
  producer?: string;
};

type CompressResult = {
  bytes: Uint8Array;
  sizeBytes: number;
  sizeLabel: string;
  savingsLabel: string;
  url: string;
};

export type PdfToolHandle = {
  openFileDialog: () => void;
  exportPdf: () => void;
};

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
};

const getPdfVersion = (buffer: ArrayBuffer) => {
  const header = new TextDecoder().decode(buffer.slice(0, 64));
  const match = header.match(/%PDF-(\d\.\d)/);
  return match ? match[1] : 'Unknown';
};

const readPdfInfo = async (file: File) => {
  const buffer = await file.arrayBuffer();
  try {
    const pdfDoc = await PDFDocument.load(buffer, { ignoreEncryption: true });
    return {
      name: file.name,
      sizeBytes: file.size,
      sizeLabel: formatBytes(file.size),
      pages: pdfDoc.getPageCount(),
      version: getPdfVersion(buffer),
      title: pdfDoc.getTitle(),
      author: pdfDoc.getAuthor(),
      producer: pdfDoc.getProducer()
    };
  } catch (error) {
    console.warn('Describe PDF failed, using basic info.', error);
    return {
      name: file.name,
      sizeBytes: file.size,
      sizeLabel: formatBytes(file.size),
      pages: null,
      version: getPdfVersion(buffer)
    };
  }
};

const repackPdf = async (buffer: ArrayBuffer, extraAggressive: boolean) => {
  const pdfDoc = await PDFDocument.load(buffer);
  if (extraAggressive) {
    pdfDoc.setTitle('');
    pdfDoc.setAuthor('');
    pdfDoc.setProducer('');
  }
  return pdfDoc.save({ useObjectStreams: true });
};

const PdfTool = forwardRef<PdfToolHandle>((_, ref) => {
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [info, setInfo] = useState<PdfInfo | null>(null);
  const [infoStatus, setInfoStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle');
  const [compressStatus, setCompressStatus] = useState<'idle' | 'working' | 'done' | 'error'>('idle');
  const [compressResult, setCompressResult] = useState<CompressResult | null>(null);
  const [isWorking, setIsWorking] = useState(false);
  const [extraAggressive, setExtraAggressive] = useState(false);
  const [previewMode, setPreviewMode] = useState<'original' | 'compressed'>('original');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fileSummary = useMemo(() => {
    if (!file) return null;
    return {
      name: file.name,
      sizeLabel: formatBytes(file.size)
    };
  }, [file]);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setFileUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    return () => {
      if (compressResult?.url) URL.revokeObjectURL(compressResult.url);
    };
  }, [compressResult]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;

    setFile(nextFile);
    setInfo(null);
    setInfoStatus('idle');
    setCompressStatus('idle');
    setCompressResult((prev) => {
      if (prev?.url) URL.revokeObjectURL(prev.url);
      return null;
    });
    setIsWorking(false);
    setPreviewMode('original');
    event.target.value = '';
  };

  const describePdf = async () => {
    if (!file) return;
    setInfoStatus('loading');
    try {
      const nextInfo = await readPdfInfo(file);
      setInfo(nextInfo);
      setInfoStatus('done');
    } catch (error) {
      console.error('Describe PDF failed', error);
      setInfoStatus('error');
    }
  };

  const compressPdf = async () => {
    if (!file) return;
    setCompressStatus('working');
    setIsWorking(true);

    try {
      const buffer = await file.arrayBuffer();
      const bytes = await repackPdf(buffer, extraAggressive);

      const blob = new Blob([bytes], { type: 'application/pdf' });
      const url = URL.createObjectURL(blob);
      const compressedSize = bytes.byteLength;
      const savingsPercent = ((1 - compressedSize / file.size) * 100).toFixed(1);

      setCompressResult({
        bytes,
        sizeBytes: compressedSize,
        sizeLabel: formatBytes(compressedSize),
        savingsLabel: `${savingsPercent}%`,
        url
      });
      setCompressStatus('done');
      setPreviewMode('compressed');
    } catch (error) {
      console.error('Compress PDF failed', error);
      setCompressStatus('error');
    } finally {
      setIsWorking(false);
    }
  };

  const downloadPdf = () => {
    if (!file && !compressResult) return;
    const url = compressResult?.url ?? (file ? URL.createObjectURL(file) : null);
    if (!url) return;

    const baseName = file?.name ? file.name.replace(/\.pdf$/i, '') : 'document';
    const link = document.createElement('a');
    link.href = url;
    link.download = compressResult ? `${baseName}-compressed.pdf` : file?.name ?? 'document.pdf';
    link.click();

    if (!compressResult && file) {
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }
  };

  useImperativeHandle(ref, () => ({
    openFileDialog: () => fileInputRef.current?.click(),
    exportPdf: downloadPdf
  }));

  return (
    <div className="flex flex-col flex-1 bg-slate-100 text-slate-900 overflow-hidden select-none">
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        accept="application/pdf"
        onChange={handleFileUpload}
      />

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 bg-white border-r border-slate-200 flex flex-col shadow-inner z-10">
          <div className="p-4 border-b border-slate-100 bg-slate-50/50">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
              <FileText size={14} /> PDF Describe &amp; Compress
            </h2>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Upload PDF</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="text-xs font-semibold text-indigo-600 hover:text-indigo-700"
                >
                  Browse
                </button>
              </div>
              {fileSummary ? (
                <div className="rounded-xl bg-slate-50 border border-slate-200 p-3">
                  <p className="text-sm font-semibold text-slate-800 truncate">{fileSummary.name}</p>
                  <p className="text-xs text-slate-500">{fileSummary.sizeLabel}</p>
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-center">
                  <Upload className="mx-auto text-slate-300" size={22} />
                  <p className="text-xs text-slate-400 mt-2">Drop a PDF or use the button above.</p>
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Describe</p>
                <button
                  onClick={describePdf}
                  disabled={!file || infoStatus === 'loading'}
                  className="text-xs font-semibold text-indigo-600 hover:text-indigo-700 disabled:text-slate-300"
                >
                  {infoStatus === 'loading' ? 'Scanning…' : 'Describe PDF'}
                </button>
              </div>
              {info ? (
                <div className="space-y-2 text-xs text-slate-600">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Pages</span>
                    <span className="font-semibold text-slate-700">{info.pages ?? 'Unknown'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Version</span>
                    <span className="font-semibold text-slate-700">{info.version}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Size</span>
                    <span className="font-semibold text-slate-700">{info.sizeLabel}</span>
                  </div>
                  {info.title && (
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Title</span>
                      <span className="font-semibold text-slate-700 truncate ml-4">{info.title}</span>
                    </div>
                  )}
                  {info.author && (
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Author</span>
                      <span className="font-semibold text-slate-700 truncate ml-4">{info.author}</span>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-xs text-slate-400">Generate a quick summary of the PDF metadata and pages.</p>
              )}
              {infoStatus === 'error' && <p className="text-xs text-red-500">Unable to read the PDF metadata.</p>}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Compress</p>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider">Repacked</span>
              </div>
              <label className="flex items-center justify-between text-xs text-slate-600">
                Extra aggressive
                <input
                  type="checkbox"
                  checked={extraAggressive}
                  onChange={(event) => setExtraAggressive(event.target.checked)}
                  className="accent-indigo-600"
                />
              </label>
              <p className="text-[11px] text-slate-400">
                Compression rewrites the PDF structure. Extra aggressive clears metadata.
              </p>
              <button
                onClick={compressPdf}
                disabled={!file || compressStatus === 'working'}
                className="w-full flex items-center justify-center gap-2 bg-slate-900 text-white px-4 py-2 rounded-lg text-xs font-semibold hover:bg-slate-800 disabled:bg-slate-300 disabled:text-slate-500"
              >
                <Sparkles size={14} />
                {compressStatus === 'working' ? 'Compressing…' : 'Compress PDF'}
              </button>
              {compressResult && (
                <button
                  onClick={downloadPdf}
                  className="w-full flex items-center justify-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg text-xs font-semibold hover:bg-indigo-500"
                >
                  <FileDown size={14} />
                  Download Compressed
                </button>
              )}
              {compressStatus === 'error' && <p className="text-xs text-red-500">Compression failed. Try again.</p>}
            </div>
          </div>
        </aside>

        <main className="flex-1 overflow-auto bg-slate-200/50 p-10 flex flex-col gap-6">
          <div className="bg-white rounded-3xl shadow-xl border border-slate-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Preview</p>
                <p className="text-sm font-semibold text-slate-800">PDF Canvas</p>
              </div>
              <div className="flex items-center gap-4">
                {compressResult && (
                  <div className="flex items-center gap-1 bg-slate-100 rounded-full p-1 text-[11px] font-semibold">
                    <button
                      onClick={() => setPreviewMode('original')}
                      className={`px-3 py-1 rounded-full transition-all ${
                        previewMode === 'original' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'
                      }`}
                    >
                      Original
                    </button>
                    <button
                      onClick={() => setPreviewMode('compressed')}
                      className={`px-3 py-1 rounded-full transition-all ${
                        previewMode === 'compressed' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'
                      }`}
                    >
                      Compressed
                    </button>
                  </div>
                )}
                {fileSummary && (
                  <div className="text-right text-xs text-slate-500">
                    <p className="font-semibold text-slate-700">{fileSummary.name}</p>
                    <p>{fileSummary.sizeLabel}</p>
                  </div>
                )}
              </div>
            </div>
            {fileUrl ? (
              <div className="h-[520px] bg-slate-50">
                <iframe
                  title="PDF preview"
                  src={previewMode === 'compressed' ? compressResult?.url ?? fileUrl : fileUrl}
                  className="w-full h-full"
                  data-testid="pdf-preview"
                />
              </div>
            ) : (
              <div className="h-[520px] flex items-center justify-center text-center text-slate-400">
                <div>
                  <FileText className="mx-auto mb-2" size={32} />
                  <p className="text-sm font-medium">No PDF loaded</p>
                  <p className="text-xs">Upload a document to preview and compress.</p>
                </div>
              </div>
            )}
          </div>

          <div className="bg-white rounded-3xl shadow-xl border border-slate-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Compression Report</p>
                <p className="text-sm font-semibold text-slate-800">Output Summary</p>
              </div>
              {isWorking && <span className="text-xs text-slate-500">Repacking file…</span>}
            </div>
            {compressResult ? (
              <div className="mt-4 grid grid-cols-3 gap-4 text-xs text-slate-600">
                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-slate-400 uppercase tracking-wider text-[10px]">Original</p>
                  <p className="text-lg font-semibold text-slate-800">{formatBytes(file?.size ?? 0)}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-slate-400 uppercase tracking-wider text-[10px]">Compressed</p>
                  <p className="text-lg font-semibold text-slate-800">{compressResult.sizeLabel}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-slate-400 uppercase tracking-wider text-[10px]">Savings</p>
                  <p className="text-lg font-semibold text-emerald-600">{compressResult.savingsLabel}</p>
                </div>
              </div>
            ) : (
              <p className="mt-4 text-xs text-slate-400">
                Compress a PDF to generate a size report and download link.
              </p>
            )}
          </div>
        </main>
      </div>
    </div>
  );
});

PdfTool.displayName = 'PdfTool';

export default PdfTool;
