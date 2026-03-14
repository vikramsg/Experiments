import { clampNotesWidth, computeSplitLayout } from '../shared/split-layout'
import type { WorkspaceSnapshot } from './note-store'

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

export type WorkspaceViews = {
  notesView: ViewLike
  splitterView: ViewLike
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

    this.views.browserView.setBounds({
      x: layout.notesWidth + SPLITTER_WIDTH,
      y: 0,
      width: layout.browserWidth,
      height: bounds.height,
    })
  }

  private publishState() {
    this.window.webContents.send('workspace:state', this.snapshot)
  }
}

export { computeSplitLayout }
