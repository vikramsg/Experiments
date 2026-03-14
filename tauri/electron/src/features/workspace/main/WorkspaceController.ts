import { IPC_CHANNELS } from '../../../ipc'
import type { WorkspaceSnapshot } from '../../../workspace-model'
import { clampNotesWidth, computeSplitLayout } from '../shared/split-layout'

export type Rectangle = {
  x: number
  y: number
  width: number
  height: number
}

type ResizeListener = () => void

export type WorkspaceWindowLike = {
  getContentBounds: () => { width: number; height: number }
  on: (event: 'resize' | 'closed', listener: ResizeListener) => void
  webContents: {
    send: (channel: string, payload: WorkspaceSnapshot) => void
  }
}

export type ViewLike = {
  setBounds: (bounds: Rectangle) => void
  webContents: {
    close: () => void
  }
}

const SPLITTER_WIDTH = 12
const MIN_NOTES_WIDTH = 280
const MIN_BROWSER_WIDTH = 360
const BROWSER_CHROME_HEIGHT = 88

export type WorkspaceViews = {
  notesView: ViewLike
  splitterView: ViewLike
  browserChromeView: ViewLike
  browserView: ViewLike
}

export class WorkspaceController {
  private snapshot: WorkspaceSnapshot

  constructor(
    private readonly window: WorkspaceWindowLike,
    private readonly views: WorkspaceViews,
    snapshot: WorkspaceSnapshot,
  ) {
    this.snapshot = snapshot

    this.window.on('resize', () => {
      this.applyLayout()
    })

    this.window.on('closed', () => {
      this.views.notesView.webContents.close()
      this.views.splitterView.webContents.close()
      this.views.browserChromeView.webContents.close()
      this.views.browserView.webContents.close()
    })

    this.applyLayout()
    this.publishState()
  }

  setNotesWidth(notesWidth: number): void {
    const bounds = this.window.getContentBounds()
    this.snapshot = {
      ...this.snapshot,
      notesWidth: clampNotesWidth({
        windowWidth: bounds.width,
        windowHeight: bounds.height,
        notesWidth,
        splitterWidth: SPLITTER_WIDTH,
        minNotesWidth: MIN_NOTES_WIDTH,
        minBrowserWidth: MIN_BROWSER_WIDTH,
      }),
    }

    this.applyLayout()
    this.publishState()
  }

  setNotes(notes: string): void {
    this.snapshot = { ...this.snapshot, notes }
    this.publishState()
  }

  setBrowserUrl(browserUrl: string): void {
    this.snapshot = { ...this.snapshot, browserUrl }
    this.publishState()
  }

  setBrowserNavigationState(input: Pick<WorkspaceSnapshot, 'browserUrl' | 'canGoBack' | 'canGoForward'>): void {
    this.snapshot = {
      ...this.snapshot,
      ...input,
    }
    this.publishState()
  }

  getSnapshot(): WorkspaceSnapshot {
    return this.snapshot
  }

  private applyLayout() {
    const bounds = this.window.getContentBounds()
    const layout = computeSplitLayout({
      windowWidth: bounds.width,
      windowHeight: bounds.height,
      notesWidth: this.snapshot.notesWidth,
      splitterWidth: SPLITTER_WIDTH,
      minNotesWidth: MIN_NOTES_WIDTH,
      minBrowserWidth: MIN_BROWSER_WIDTH,
    })

    this.snapshot = {
      ...this.snapshot,
      notesWidth: layout.notesWidth,
    }

    const rightPaneX = layout.notesWidth + SPLITTER_WIDTH
    const browserChromeHeight = Math.min(BROWSER_CHROME_HEIGHT, bounds.height)
    const browserContentHeight = Math.max(0, bounds.height - browserChromeHeight)

    this.views.notesView.setBounds({
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
      height: browserContentHeight,
    })
  }

  private publishState() {
    this.window.webContents.send(IPC_CHANNELS.workspaceState, this.snapshot)
  }
}

export { BROWSER_CHROME_HEIGHT, computeSplitLayout }
