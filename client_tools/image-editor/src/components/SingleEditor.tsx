import React, { forwardRef, useImperativeHandle, useRef, useState } from 'react';
import { Check, Crop, RefreshCw, Upload } from 'lucide-react';
import type { ActiveHandle, ImageLayer, Point, Size } from '../types';

export type SingleEditorHandle = {
  openFileDialog: () => void;
  exportImage: () => void;
};

const MAX_INITIAL_WIDTH = 800;

const SingleEditor = forwardRef<SingleEditorHandle>((_, ref) => {
  const [singleImage, setSingleImage] = useState<ImageLayer | null>(null);
  const [workspaceSize, setWorkspaceSize] = useState<Size>({ width: 800, height: 600 });
  const [isDragging, setIsDragging] = useState(false);
  const [activeHandle, setActiveHandle] = useState<ActiveHandle | null>(null);
  const [dragStart, setDragStart] = useState<Point>({ x: 0, y: 0 });
  const [initialState, setInitialState] = useState<ImageLayer | null>(null);

  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const getWorkspacePoint = (event: React.MouseEvent) => {
    if (!workspaceRef.current) return null;
    const rect = workspaceRef.current.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (loadEvent) => {
      const result = loadEvent.target?.result;
      if (typeof result !== 'string') return;

      const img = new Image();
      img.onload = () => {
        let initWidth = img.width;
        let initHeight = img.height;

        if (img.width > MAX_INITIAL_WIDTH) {
          const scale = MAX_INITIAL_WIDTH / img.width;
          initWidth = MAX_INITIAL_WIDTH;
          initHeight = img.height * scale;
        }

        const newImage: ImageLayer = {
          id: Date.now() + Math.random(),
          name: file.name,
          src: result,
          naturalWidth: img.width,
          naturalHeight: img.height,
          x: 0,
          y: 0,
          width: initWidth,
          height: initHeight,
          contentWidth: initWidth,
          contentHeight: initHeight,
          contentX: 0,
          contentY: 0,
          zIndex: 0,
          isBase: true,
          rotation: 0
        };

        setWorkspaceSize({ width: initWidth, height: initHeight });
        setSingleImage(newImage);
      };
      img.src = result;
    };
    reader.readAsDataURL(file);

    event.target.value = '';
  };

  const handleMouseDown = (event: React.MouseEvent, handle: ActiveHandle) => {
    event.stopPropagation();
    event.preventDefault();

    if (!singleImage) return;

    const point = getWorkspacePoint(event);
    if (!point) return;

    setIsDragging(true);
    setActiveHandle(handle);
    setDragStart(point);
    setInitialState({ ...singleImage });
  };

  const handleMouseMove = (event: React.MouseEvent) => {
    if (!isDragging || !initialState || !activeHandle || !singleImage) return;
    const point = getWorkspacePoint(event);
    if (!point) return;

    const dx = point.x - dragStart.x;
    const dy = point.y - dragStart.y;

    if (activeHandle === 'pan-content') {
      setSingleImage({
        ...singleImage,
        contentX: initialState.contentX + dx,
        contentY: initialState.contentY + dy
      });
      return;
    }

    let newX = initialState.x;
    let newY = initialState.y;
    let newW = initialState.width;
    let newH = initialState.height;
    let newContentX = initialState.contentX;
    let newContentY = initialState.contentY;

    if (activeHandle.includes('e')) newW = Math.max(10, initialState.width + dx);
    if (activeHandle.includes('s')) newH = Math.max(10, initialState.height + dy);

    if (activeHandle.includes('w')) {
      newW = Math.max(10, initialState.width - dx);
      newX = initialState.x + (initialState.width - newW);
      newContentX = initialState.contentX - (newX - initialState.x);
    }

    if (activeHandle.includes('n')) {
      newH = Math.max(10, initialState.height - dy);
      newY = initialState.y + (initialState.height - newH);
      newContentY = initialState.contentY - (newY - initialState.y);
    }

    setSingleImage({
      ...singleImage,
      x: newX,
      y: newY,
      width: newW,
      height: newH,
      contentX: newContentX,
      contentY: newContentY
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setActiveHandle(null);
    setInitialState(null);
  };

  const resetCrop = () => {
    if (!singleImage) return;
    setSingleImage({
      ...singleImage,
      width: singleImage.contentWidth,
      height: singleImage.contentHeight,
      contentX: 0,
      contentY: 0
    });
  };

  const exportImage = () => {
    if (!singleImage) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = singleImage.width * 2;
    canvas.height = singleImage.height * 2;
    ctx.scale(2, 2);

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, singleImage.width, singleImage.height);

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, singleImage.width, singleImage.height);
      ctx.clip();

      ctx.drawImage(
        img,
        singleImage.contentX,
        singleImage.contentY,
        singleImage.contentWidth,
        singleImage.contentHeight
      );

      ctx.restore();

      const link = document.createElement('a');
      link.download = `crop-${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };
    img.src = singleImage.src;
  };

  useImperativeHandle(ref, () => ({
    openFileDialog: () => fileInputRef.current?.click(),
    exportImage
  }));

  return (
    <div
      className="flex flex-col flex-1 bg-slate-100 text-slate-900 overflow-hidden select-none"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        accept="image/*"
        onChange={handleFileUpload}
      />

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-72 bg-white border-r border-slate-200 flex flex-col shadow-inner z-10">
          <div className="p-4 border-b border-slate-100 bg-slate-50/50">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
              <Crop size={14} /> Single Image Crop
            </h2>
          </div>

          <div className="flex-1 p-4 space-y-4">
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-2">Workflow</p>
              <p className="text-sm text-slate-600">Upload a single image, drag handles to crop, and pan the image inside the frame.</p>
            </div>
            <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-center">
              <Upload className="mx-auto text-slate-300" size={22} />
              <p className="text-xs text-slate-400 mt-2">Use the Add Image button above.</p>
            </div>
          </div>
        </aside>

        <main className="flex-1 relative overflow-auto bg-slate-200/50 p-16 flex justify-center items-start scrollbar-hide">
          <div
            ref={workspaceRef}
            data-testid="single-workspace"
            className="relative bg-white shadow-2xl transition-all duration-300 ease-out"
            style={{
              width: `${workspaceSize.width}px`,
              height: `${workspaceSize.height}px`
            }}
          >
            <div
              className="absolute inset-0 opacity-[0.03] pointer-events-none"
              style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 0)', backgroundSize: '20px 20px' }}
            />

            {singleImage ? (
              <div
                style={{
                  position: 'absolute',
                  left: `${singleImage.x}px`,
                  top: `${singleImage.y}px`,
                  width: `${singleImage.width}px`,
                  height: `${singleImage.height}px`
                }}
              >
                <div className="w-full h-full relative overflow-hidden pointer-events-none select-none outline outline-1 outline-slate-200">
                  <img
                    src={singleImage.src}
                    alt=""
                    className="absolute max-w-none"
                    style={{
                      width: `${singleImage.contentWidth}px`,
                      height: `${singleImage.contentHeight}px`,
                      left: `${singleImage.contentX}px`,
                      top: `${singleImage.contentY}px`
                    }}
                  />
                </div>

                <div
                  className="absolute pointer-events-none opacity-30 grayscale"
                  style={{
                    width: `${singleImage.contentWidth}px`,
                    height: `${singleImage.contentHeight}px`,
                    left: `${singleImage.contentX}px`,
                    top: `${singleImage.contentY}px`,
                    zIndex: -1,
                    outline: '2px dashed #6366f1'
                  }}
                >
                  <img src={singleImage.src} alt="" className="w-full h-full" />
                </div>

                <div className="absolute inset-0 pointer-events-auto">
                  <div className="absolute inset-0 ring-2 ring-indigo-500 ring-offset-0 pointer-events-none" />

                  <div
                    className="absolute -top-1.5 -left-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-nw-resize"
                    onMouseDown={(event) => handleMouseDown(event, 'nw')}
                  />
                  <div
                    className="absolute -top-1.5 -right-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-ne-resize"
                    onMouseDown={(event) => handleMouseDown(event, 'ne')}
                  />
                  <div
                    className="absolute -bottom-1.5 -left-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-sw-resize"
                    onMouseDown={(event) => handleMouseDown(event, 'sw')}
                  />
                  <div
                    className="absolute -bottom-1.5 -right-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-se-resize"
                    onMouseDown={(event) => handleMouseDown(event, 'se')}
                  />

                  <div
                    className="absolute top-1/2 -left-1 -translate-y-1/2 w-2 h-8 bg-indigo-600 cursor-w-resize rounded-l shadow-sm"
                    onMouseDown={(event) => handleMouseDown(event, 'w')}
                  />
                  <div
                    className="absolute top-1/2 -right-1 -translate-y-1/2 w-2 h-8 bg-indigo-600 cursor-e-resize rounded-r shadow-sm"
                    onMouseDown={(event) => handleMouseDown(event, 'e')}
                  />
                  <div
                    className="absolute -top-1 left-1/2 -translate-x-1/2 w-8 h-2 bg-indigo-600 cursor-n-resize rounded-t shadow-sm"
                    onMouseDown={(event) => handleMouseDown(event, 'n')}
                  />
                  <div
                    className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-8 h-2 bg-indigo-600 cursor-s-resize rounded-b shadow-sm"
                    onMouseDown={(event) => handleMouseDown(event, 's')}
                  />

                  <div
                    className="absolute inset-4 cursor-move hover:bg-indigo-500/10 transition-colors rounded"
                    title="Drag to pan image inside frame"
                    onMouseDown={(event) => handleMouseDown(event, 'pan-content')}
                  />

                  <div className="absolute -bottom-14 left-1/2 -translate-x-1/2 flex gap-1 bg-slate-900 text-white p-1.5 rounded-lg shadow-xl z-50">
                    <button
                      onClick={(event) => {
                        event.stopPropagation();
                      }}
                      className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 rounded flex items-center gap-2 text-xs font-bold transition-colors"
                    >
                      <Check size={14} /> CROP MODE
                    </button>
                    <div className="w-px h-4 bg-slate-700 self-center mx-1" />
                    <button
                      onClick={(event) => {
                        event.stopPropagation();
                        resetCrop();
                      }}
                      className="p-1 hover:bg-slate-700 rounded text-slate-400 hover:text-white transition-colors"
                      title="Reset Mask"
                    >
                      <RefreshCw size={14} />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center py-12 px-6 border-2 border-dashed border-slate-200 rounded-2xl bg-white/70">
                  <Upload className="mx-auto text-slate-300 mb-2" size={32} />
                  <p className="text-xs text-slate-400 font-medium">Upload a single image to start cropping</p>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      <footer className="bg-slate-900 text-slate-400 px-6 py-2.5 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest border-t border-slate-800">
        <div className="flex gap-6">
          <span className="text-indigo-400 flex items-center gap-2">
            <Crop size={12} /> SINGLE CROP MODE - DRAG HANDLES TO MASK, CENTER TO PAN
          </span>
        </div>
        <div className="text-slate-600">Single Image Cropper v1.0</div>
      </footer>
    </div>
  );
});

SingleEditor.displayName = 'SingleEditor';

export default SingleEditor;
