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

export type BrowserViewLike = {
  setBounds: (bounds: Rectangle) => void
  webContents: {
    close: () => void
  }
}

const HEADER_HEIGHT = 72
const SPLITTER_WIDTH = 12
const MIN_NOTES_WIDTH = 280
const MIN_BROWSER_WIDTH = 360

export class WorkspaceController {
  private snapshot: WorkspaceSnapshot

  constructor(
    private readonly window: WorkspaceWindowLike,
    private readonly browserView: BrowserViewLike,
    snapshot: WorkspaceSnapshot,
  ) {
    this.snapshot = snapshot

    this.window.on('resize', () => {
      this.applyLayout()
    })

    this.window.on('closed', () => {
      this.browserView.webContents.close()
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

    this.browserView.setBounds({
      x: layout.notesWidth + SPLITTER_WIDTH,
      y: HEADER_HEIGHT,
      width: layout.browserWidth,
      height: Math.max(0, bounds.height - HEADER_HEIGHT),
    })
  }

  private publishState() {
    this.window.webContents.send('workspace:state', this.snapshot)
  }
}

export { computeSplitLayout }
