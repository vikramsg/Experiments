import type { WorkspaceSnapshot } from './main/note-store'

export type LauncherApi = {
  openWorkspace: () => Promise<void>
}

export type WorkspaceApi = {
  loadState: () => Promise<WorkspaceSnapshot>
  saveNotes: (notes: string) => Promise<void>
  setBrowserUrl: (url: string) => Promise<void>
  adjustSplitter: (delta: number) => Promise<void>
  onStateChange: (listener: (snapshot: WorkspaceSnapshot) => void) => () => void
}

declare global {
  interface Window {
    launcher: LauncherApi
    workspace: WorkspaceApi
  }
}

export {}
