import React, { forwardRef, useImperativeHandle, useRef, useState } from 'react';
import { Anchor, Check, Crop, Image as ImageIcon, Layers, MousePointer2, RefreshCw, Trash2 } from 'lucide-react';
import type { ActiveHandle, HandleAction, ImageLayer, InteractionMode, Point, Size } from '../types';

export type CompositorEditorHandle = {
  openFileDialog: () => void;
  exportImage: () => void;
};

const MAX_INITIAL_WIDTH = 800;

const CompositorEditor = forwardRef<CompositorEditorHandle>((_, ref) => {
  const [images, setImages] = useState<ImageLayer[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('select');
  const [isDragging, setIsDragging] = useState(false);
  const [activeHandle, setActiveHandle] = useState<ActiveHandle | null>(null);
  const [dragStart, setDragStart] = useState<Point>({ x: 0, y: 0 });
  const [initialState, setInitialState] = useState<ImageLayer | null>(null);
  const [workspaceSize, setWorkspaceSize] = useState<Size>({ width: 800, height: 600 });

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
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) return;

    files.forEach((file) => {
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

          const id = Date.now() + Math.random();

          setImages((prev) => {
            const isFirst = prev.length === 0;

            if (isFirst) {
              setWorkspaceSize({ width: initWidth, height: initHeight });
            }

            const newImage: ImageLayer = {
              id,
              name: file.name,
              src: result,
              naturalWidth: img.width,
              naturalHeight: img.height,
              x: isFirst ? 0 : 50,
              y: isFirst ? 0 : 50,
              width: initWidth,
              height: initHeight,
              contentWidth: initWidth,
              contentHeight: initHeight,
              contentX: 0,
              contentY: 0,
              zIndex: prev.length + 1,
              isBase: isFirst,
              rotation: 0
            };

            setSelectedId(newImage.id);
            setInteractionMode('select');

            return [...prev, newImage];
          });
        };
        img.src = result;
      };
      reader.readAsDataURL(file);
    });

    event.target.value = '';
  };

  const setAsBase = (id: number) => {
    setImages((prev) => {
      const base = prev.find((img) => img.id === id);
      if (!base) return prev;

      setWorkspaceSize({ width: base.width, height: base.height });

      return prev.map((img) => {
        if (img.id === id) {
          return { ...img, isBase: true, x: 0, y: 0, zIndex: 0 };
        }
        return { ...img, isBase: false };
      });
    });
  };

  const handleMouseDown = (event: React.MouseEvent, id: number, handle: HandleAction = null) => {
    event.stopPropagation();
    event.preventDefault();

    if (selectedId !== id) {
      setSelectedId(id);
      setInteractionMode('select');
    }

    const img = images.find((layer) => layer.id === id);
    if (!img) return;

    if (handle === 'activate-crop') {
      setInteractionMode('crop');
      return;
    }

    if (img.isBase && handle !== 'select' && interactionMode !== 'crop') return;

    const point = getWorkspacePoint(event);
    if (!point) return;

    setIsDragging(true);
    setActiveHandle(handle && handle !== 'select' ? handle : 'move');
    setDragStart(point);
    setInitialState({ ...img });
  };

  const handleMouseMove = (event: React.MouseEvent) => {
    if (!isDragging || !initialState || !activeHandle) return;
    const point = getWorkspacePoint(event);
    if (!point) return;

    const dx = point.x - dragStart.x;
    const dy = point.y - dragStart.y;

    setImages((prev) =>
      prev.map((img) => {
        if (img.id !== selectedId) return img;

        if (activeHandle === 'move') {
          return {
            ...img,
            x: initialState.x + dx,
            y: initialState.y + dy
          };
        }

        if (interactionMode === 'select' && ['nw', 'ne', 'sw', 'se'].includes(activeHandle)) {
          const aspect = initialState.width / initialState.height;
          let newW = initialState.width;
          let newH = initialState.height;
          let newX = initialState.x;
          let newY = initialState.y;

          if (activeHandle.includes('e')) newW = Math.max(20, initialState.width + dx);
          if (activeHandle.includes('w')) {
            newW = Math.max(20, initialState.width - dx);
            newX = initialState.x + (initialState.width - newW);
          }

          newH = newW / aspect;
          if (activeHandle.includes('n')) newY = initialState.y + (initialState.height - newH);

          const scaleFactor = newW / initialState.width;

          return {
            ...img,
            x: newX,
            y: newY,
            width: newW,
            height: newH,
            contentWidth: initialState.contentWidth * scaleFactor,
            contentHeight: initialState.contentHeight * scaleFactor,
            contentX: initialState.contentX * scaleFactor,
            contentY: initialState.contentY * scaleFactor
          };
        }

        if (interactionMode === 'crop') {
          if (activeHandle === 'pan-content') {
            return {
              ...img,
              contentX: initialState.contentX + dx,
              contentY: initialState.contentY + dy
            };
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

          return {
            ...img,
            x: newX,
            y: newY,
            width: newW,
            height: newH,
            contentX: newContentX,
            contentY: newContentY
          };
        }

        return img;
      })
    );
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setActiveHandle(null);
    setInitialState(null);
  };

  const deleteImage = (id: number) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
    if (selectedId === id) {
      setSelectedId(null);
      setInteractionMode('select');
    }
  };

  const bringToFront = (id: number) => {
    setImages((prev) => {
      const maxZ = Math.max(...prev.map((img) => img.zIndex), 0);
      return prev.map((img) => (img.id === id ? { ...img, zIndex: maxZ + 1, isBase: false } : img));
    });
  };

  const resetCrop = (id: number) => {
    setImages((prev) =>
      prev.map((img) => {
        if (img.id !== id) return img;
        return {
          ...img,
          width: img.contentWidth,
          height: img.contentHeight,
          contentX: 0,
          contentY: 0
        };
      })
    );
  };

  const exportImage = () => {
    if (images.length === 0) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = workspaceSize.width * 2;
    canvas.height = workspaceSize.height * 2;
    ctx.scale(2, 2);

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, workspaceSize.width, workspaceSize.height);

    const sortedImages = [...images].sort((a, b) => a.zIndex - b.zIndex);

    let loadedCount = 0;
    sortedImages.forEach((imgData) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        ctx.save();
        ctx.beginPath();
        ctx.rect(imgData.x, imgData.y, imgData.width, imgData.height);
        ctx.clip();

        ctx.drawImage(
          img,
          imgData.x + imgData.contentX,
          imgData.y + imgData.contentY,
          imgData.contentWidth,
          imgData.contentHeight
        );

        ctx.restore();

        loadedCount += 1;
        if (loadedCount === sortedImages.length) {
          const link = document.createElement('a');
          link.download = `composition-${Date.now()}.png`;
          link.href = canvas.toDataURL('image/png');
          link.click();
        }
      };
      img.src = imgData.src;
    });
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
        multiple
        accept="image/*"
        onChange={handleFileUpload}
      />

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-72 bg-white border-r border-slate-200 flex flex-col shadow-inner z-10">
          <div className="p-4 border-b border-slate-100 bg-slate-50/50">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
              <ImageIcon size={14} /> Layers Stack
            </h2>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {images.length === 0 ? (
              <div className="text-center py-12 px-4 border-2 border-dashed border-slate-200 rounded-2xl">
                <ImageIcon className="mx-auto text-slate-300 mb-2" size={32} />
                <p className="text-xs text-slate-400 font-medium">Upload images to begin</p>
              </div>
            ) : (
              [...images]
                .sort((a, b) => b.zIndex - a.zIndex)
                .map((img) => (
                  <div
                    key={img.id}
                    onClick={() => {
                      setSelectedId(img.id);
                      setInteractionMode('select');
                    }}
                    className={`group relative p-3 rounded-xl border transition-all cursor-pointer ${
                      selectedId === img.id
                        ? 'border-indigo-500 bg-indigo-50 shadow-sm'
                        : 'border-slate-100 bg-white hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-lg overflow-hidden bg-slate-100 border border-slate-200 flex items-center justify-center shrink-0">
                        <img src={img.src} alt="" className="max-w-full max-h-full object-cover" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-[11px] font-bold text-slate-700 truncate mb-1">
                          {img.isBase ? 'BASE LAYER' : img.name.split('.')[0]}
                        </p>
                        <div className="flex gap-2">
                          <button
                            onClick={(event) => {
                              event.stopPropagation();
                              bringToFront(img.id);
                            }}
                            className="p-1 text-slate-400 hover:text-indigo-600 transition-colors"
                            title="Bring to Front"
                          >
                            <Layers size={14} />
                          </button>
                          {!img.isBase && (
                            <>
                              <button
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setAsBase(img.id);
                                }}
                                className="p-1 text-slate-400 hover:text-emerald-600 transition-colors"
                                title="Set as Background"
                              >
                                <Anchor size={14} />
                              </button>
                              <button
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setSelectedId(img.id);
                                  setInteractionMode(
                                    interactionMode === 'crop' && selectedId === img.id ? 'select' : 'crop'
                                  );
                                }}
                                className={`p-1 transition-colors ${
                                  interactionMode === 'crop' && selectedId === img.id
                                    ? 'text-indigo-600'
                                    : 'text-slate-400 hover:text-indigo-600'
                                }`}
                                title="Crop Tool"
                              >
                                <Crop size={14} />
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(event) => {
                          event.stopPropagation();
                          deleteImage(img.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-2 text-slate-400 hover:text-red-500 transition-all"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                ))
            )}
          </div>

          <div className="p-4 border-t border-slate-100 bg-slate-50">
            <div className="bg-white rounded-lg p-3 border border-slate-200">
              <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Instructions</p>
              <p className="text-xs text-slate-600 mb-1">Double-click image to Crop.</p>
              <p className="text-xs text-slate-600">Drag edges to mask.</p>
            </div>
          </div>
        </aside>

        <main className="flex-1 relative overflow-auto bg-slate-200/50 p-16 flex justify-center items-start scrollbar-hide">
          <div
            ref={workspaceRef}
            data-testid="compositor-workspace"
            className="relative bg-white shadow-2xl transition-all duration-300 ease-out"
            style={{
              width: `${workspaceSize.width}px`,
              height: `${workspaceSize.height}px`
            }}
            onClick={() => {
              setSelectedId(null);
              setInteractionMode('select');
            }}
          >
            <div
              className="absolute inset-0 opacity-[0.03] pointer-events-none"
              style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 0)', backgroundSize: '20px 20px' }}
            />

            {images.map((img) => {
              const isSelected = selectedId === img.id;
              const isCropping = isSelected && interactionMode === 'crop';

              return (
                <div
                  key={img.id}
                  style={{
                    position: 'absolute',
                    left: `${img.x}px`,
                    top: `${img.y}px`,
                    width: `${img.width}px`,
                    height: `${img.height}px`,
                    zIndex: img.zIndex
                  }}
                  className="group"
                  onMouseDown={(event) => !isCropping && handleMouseDown(event, img.id, img.isBase ? 'select' : 'move')}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    setSelectedId(img.id);
                    setInteractionMode('crop');
                  }}
                >
                  <div className="w-full h-full relative overflow-hidden pointer-events-none select-none outline outline-1 outline-slate-200">
                    <img
                      src={img.src}
                      alt=""
                      className="absolute max-w-none"
                      style={{
                        width: `${img.contentWidth}px`,
                        height: `${img.contentHeight}px`,
                        left: `${img.contentX}px`,
                        top: `${img.contentY}px`
                      }}
                    />
                  </div>

                  {isCropping && (
                    <div
                      className="absolute pointer-events-none opacity-30 grayscale"
                      style={{
                        width: `${img.contentWidth}px`,
                        height: `${img.contentHeight}px`,
                        left: `${img.contentX}px`,
                        top: `${img.contentY}px`,
                        zIndex: -1,
                        outline: '2px dashed #6366f1'
                      }}
                    >
                      <img src={img.src} alt="" className="w-full h-full" />
                    </div>
                  )}

                  {isSelected && !img.isBase && (
                    <div className="absolute inset-0 pointer-events-auto">
                      {interactionMode === 'select' && (
                        <div className="w-full h-full ring-2 ring-indigo-500 ring-offset-2 relative">
                          <div
                            className="absolute -top-1.5 -left-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-nw-resize"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'nw')}
                          />
                          <div
                            className="absolute -top-1.5 -right-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-ne-resize"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'ne')}
                          />
                          <div
                            className="absolute -bottom-1.5 -left-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-sw-resize"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'sw')}
                          />
                          <div
                            className="absolute -bottom-1.5 -right-1.5 w-3 h-3 bg-white border border-indigo-500 cursor-se-resize"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'se')}
                          />

                          <div className="absolute -top-12 left-1/2 -translate-x-1/2 flex gap-1 bg-white p-1 rounded-lg shadow-lg border border-slate-200 z-50 animate-in fade-in zoom-in duration-200">
                            <button
                              onClick={(event) => handleMouseDown(event, img.id, 'activate-crop')}
                              className="px-2 py-1 hover:bg-indigo-50 rounded text-slate-700 hover:text-indigo-600 flex items-center gap-2 text-xs font-bold"
                            >
                              <Crop size={14} /> CROP
                            </button>
                          </div>
                        </div>
                      )}

                      {interactionMode === 'crop' && (
                        <div className="w-full h-full relative">
                          <div className="absolute inset-0 ring-2 ring-indigo-500 ring-offset-0 pointer-events-none" />

                          <div
                            className="absolute top-1/2 -left-1 -translate-y-1/2 w-2 h-8 bg-indigo-600 cursor-w-resize rounded-l shadow-sm"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'w')}
                          />
                          <div
                            className="absolute top-1/2 -right-1 -translate-y-1/2 w-2 h-8 bg-indigo-600 cursor-e-resize rounded-r shadow-sm"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'e')}
                          />
                          <div
                            className="absolute -top-1 left-1/2 -translate-x-1/2 w-8 h-2 bg-indigo-600 cursor-n-resize rounded-t shadow-sm"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'n')}
                          />
                          <div
                            className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-8 h-2 bg-indigo-600 cursor-s-resize rounded-b shadow-sm"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 's')}
                          />

                          <div
                            className="absolute inset-4 cursor-move hover:bg-indigo-500/10 transition-colors rounded"
                            title="Drag to pan image inside frame"
                            onMouseDown={(event) => handleMouseDown(event, img.id, 'pan-content')}
                          />

                          <div className="absolute -bottom-14 left-1/2 -translate-x-1/2 flex gap-1 bg-slate-900 text-white p-1.5 rounded-lg shadow-xl z-50">
                            <button
                              onClick={(event) => {
                                event.stopPropagation();
                                setInteractionMode('select');
                              }}
                              className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 rounded flex items-center gap-2 text-xs font-bold transition-colors"
                            >
                              <Check size={14} /> DONE
                            </button>
                            <div className="w-px h-4 bg-slate-700 self-center mx-1" />
                            <button
                              onClick={(event) => {
                                event.stopPropagation();
                                resetCrop(img.id);
                              }}
                              className="p-1 hover:bg-slate-700 rounded text-slate-400 hover:text-white transition-colors"
                              title="Reset Mask"
                            >
                              <RefreshCw size={14} />
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {img.isBase && (
                    <div className="absolute top-2 left-2 bg-slate-800/80 text-white text-[9px] px-2.5 py-1 rounded-full font-bold tracking-widest backdrop-blur-sm pointer-events-none shadow-sm">
                      BASE DOCUMENT
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </main>
      </div>

      <footer className="bg-slate-900 text-slate-400 px-6 py-2.5 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest border-t border-slate-800">
        <div className="flex gap-6">
          {interactionMode === 'crop' ? (
            <span className="text-indigo-400 flex items-center gap-2 animate-pulse">
              <Crop size={12} /> CROP MODE ACTIVE - DRAG HANDLES TO MASK, CENTER TO PAN
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <MousePointer2 size={12} /> SELECT MODE - DRAG TO MOVE, CORNERS TO RESIZE
            </span>
          )}
        </div>
        <div className="text-slate-600">Compositor Engine v3.0 (Masking)</div>
      </footer>
    </div>
  );
});

CompositorEditor.displayName = 'CompositorEditor';

export default CompositorEditor;
