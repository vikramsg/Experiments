/**
 * Public renderer-facing contract for the Browser + Notes workspace.
 *
 * This file defines the typed API exposed through the preload bridge and used
 * by renderer features. It lives at the root of `src/` because it is a stable
 * application contract that multiple isolated features rely on. Making it a
 * root file keeps the contract easy to find and prevents the ambient global
 * declaration file from becoming the primary source of truth.
 *
 * Import guidance:
 * - renderer features may import this contract directly.
 * - preload and app composition code may import this contract directly.
 * - features must not import one another; they should communicate through this
 *   contract and the workspace model instead.
 */
import type { WorkspaceSnapshot } from './workspace-model'

export type LauncherApi = {
  openWorkspace: () => Promise<void>
  openOpenCode: () => Promise<void>
  openTerminal: () => Promise<void>
}

export type WorkspaceApi = {
  loadState: () => Promise<WorkspaceSnapshot>
  saveNotes: (notes: string) => Promise<void>
  setBrowserUrl: (url: string) => Promise<void>
  goBack: () => Promise<void>
  goForward: () => Promise<void>
  adjustSplitter: (delta: number) => Promise<void>
  onStateChange: (listener: (snapshot: WorkspaceSnapshot) => void) => () => void
}
