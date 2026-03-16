import { BROWSER_CHROME_HEIGHT } from './browser-host'
import { clampNotesWidth, computeSplitLayout } from '../../features/workspace/shared/split-layout'

type ResizeListener = () => void

export type Rectangle = {
  x: number
  y: number
  width: number
  height: number
}

export type OpenCodeBrowserWindowLike = {
  getContentBounds: () => { width: number; height: number }
  on: (event: 'resize' | 'closed', listener: ResizeListener) => void
}

export type ViewLike = {
  setBounds: (bounds: Rectangle) => void
  webContents: {
    close: () => void
  }
}

export type OpenCodeBrowserViews = {
  openCodeView: ViewLike
  splitterView: ViewLike
  browserChromeView: ViewLike
  browserView: ViewLike
}

const SPLITTER_WIDTH = 12
const MIN_OPEN_CODE_WIDTH = 320
const MIN_BROWSER_WIDTH = 420

export class OpenCodeBrowserController {
  private openCodeWidth: number

  constructor(
    private readonly window: OpenCodeBrowserWindowLike,
    private readonly views: OpenCodeBrowserViews,
    initialOpenCodeWidth: number,
  ) {
    this.openCodeWidth = initialOpenCodeWidth

    this.window.on('resize', () => {
      this.applyLayout()
    })

    this.window.on('closed', () => {
      this.views.openCodeView.webContents.close()
      this.views.splitterView.webContents.close()
      this.views.browserChromeView.webContents.close()
      this.views.browserView.webContents.close()
    })

    this.applyLayout()
  }

  adjustOpenCodeWidth(delta: number): void {
    // Reuse the shared split-layout math so the OpenCode + Browser launcher
    // follows the same safe min-width rules as Browser + Notes.
    const bounds = this.window.getContentBounds()
    this.openCodeWidth = clampNotesWidth({
      windowWidth: bounds.width,
      windowHeight: bounds.height,
      notesWidth: this.openCodeWidth + delta,
      splitterWidth: SPLITTER_WIDTH,
      minNotesWidth: MIN_OPEN_CODE_WIDTH,
      minBrowserWidth: MIN_BROWSER_WIDTH,
    })

    this.applyLayout()
  }

  getOpenCodeWidth(): number {
    return this.openCodeWidth
  }

  private applyLayout(): void {
    const bounds = this.window.getContentBounds()
    const layout = computeSplitLayout({
      windowWidth: bounds.width,
      windowHeight: bounds.height,
      notesWidth: this.openCodeWidth,
      splitterWidth: SPLITTER_WIDTH,
      minNotesWidth: MIN_OPEN_CODE_WIDTH,
      minBrowserWidth: MIN_BROWSER_WIDTH,
    })

    this.openCodeWidth = layout.notesWidth
    const rightPaneX = layout.notesWidth + SPLITTER_WIDTH
    const browserChromeHeight = Math.min(BROWSER_CHROME_HEIGHT, bounds.height)

    this.views.openCodeView.setBounds({
      x: 0,
      y: 0,
      width: layout.notesWidth,
      height: bounds.height,
    })

    this.views.splitterView.setBounds({
      x: layout.splitterX,
      y: 0,
      width: SPLITTER_WIDTH,
      height: bounds.height,
    })

    this.views.browserChromeView.setBounds({
      x: rightPaneX,
      y: 0,
      width: layout.browserWidth,
      height: browserChromeHeight,
    })

    this.views.browserView.setBounds({
      x: rightPaneX,
      y: browserChromeHeight,
      width: layout.browserWidth,
      height: Math.max(0, bounds.height - browserChromeHeight),
    })
  }
}
