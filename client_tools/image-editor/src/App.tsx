import React, { useRef, useState } from 'react';
import { Download, FileDown, Layers, Upload } from 'lucide-react';
import CompositorEditor, { type CompositorEditorHandle } from './components/CompositorEditor';
import PdfTool, { type PdfToolHandle } from './components/PdfTool';
import SingleEditor, { type SingleEditorHandle } from './components/SingleEditor';

type TabKey = 'compositor' | 'single' | 'pdf';

const App = () => {
  const [activeTab, setActiveTab] = useState<TabKey>('compositor');
  const compositorRef = useRef<CompositorEditorHandle | null>(null);
  const singleRef = useRef<SingleEditorHandle | null>(null);
  const pdfRef = useRef<PdfToolHandle | null>(null);

  const handleAddImage = () => {
    if (activeTab === 'compositor') {
      compositorRef.current?.openFileDialog();
    } else if (activeTab === 'single') {
      singleRef.current?.openFileDialog();
    } else {
      pdfRef.current?.openFileDialog();
    }
  };

  const handleExport = () => {
    if (activeTab === 'compositor') {
      compositorRef.current?.exportImage();
    } else if (activeTab === 'single') {
      singleRef.current?.exportImage();
    } else {
      pdfRef.current?.exportPdf();
    }
  };

  const primaryLabel = activeTab === 'pdf' ? 'Upload PDF' : 'Add Image';
  const exportLabel = activeTab === 'pdf' ? 'Download PDF' : 'Export';
  const exportIcon = activeTab === 'pdf' ? <FileDown size={18} /> : <Download size={18} />;

  return (
    <div className="flex flex-col h-screen bg-slate-100 font-sans text-slate-900 overflow-hidden">
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm z-20">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-lg shadow-indigo-100 shadow-lg">
            <Layers className="text-white w-5 h-5" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight">Pro Image Compositor</h1>
            <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Masking & Layout</p>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-slate-100 rounded-full p-1">
          <button
            onClick={() => setActiveTab('compositor')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
              activeTab === 'compositor'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            Multi-Layer
          </button>
          <button
            onClick={() => setActiveTab('single')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
              activeTab === 'single'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            Single Image
          </button>
          <button
            onClick={() => setActiveTab('pdf')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
              activeTab === 'pdf' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            PDF Tools
          </button>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={handleAddImage}
            className="flex items-center gap-2 bg-slate-800 text-white px-5 py-2 rounded-lg font-medium hover:bg-slate-700 transition-all active:scale-95"
          >
            <Upload size={18} />
            {primaryLabel}
          </button>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 bg-indigo-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-indigo-700 transition-all shadow-md shadow-indigo-100 active:scale-95"
          >
            {exportIcon}
            {exportLabel}
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {activeTab === 'compositor' ? (
          <CompositorEditor ref={compositorRef} />
        ) : activeTab === 'single' ? (
          <SingleEditor ref={singleRef} />
        ) : (
          <PdfTool ref={pdfRef} />
        )}
      </div>
    </div>
  );
};

export default App;
