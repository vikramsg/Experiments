export type InteractionMode = 'select' | 'crop';
export type ResizeHandle = 'nw' | 'ne' | 'sw' | 'se' | 'n' | 's' | 'e' | 'w';
export type ActiveHandle = 'move' | ResizeHandle | 'pan-content';
export type HandleAction = ActiveHandle | 'activate-crop' | 'select' | null;

export type Point = {
  x: number;
  y: number;
};

export type Size = {
  width: number;
  height: number;
};

export type ImageLayer = {
  id: number;
  name: string;
  src: string;
  naturalWidth: number;
  naturalHeight: number;
  x: number;
  y: number;
  width: number;
  height: number;
  contentWidth: number;
  contentHeight: number;
  contentX: number;
  contentY: number;
  zIndex: number;
  isBase: boolean;
  rotation: number;
};
