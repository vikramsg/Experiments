import type { LauncherApi, WorkspaceApi } from './workspace-contract'

/**
 * Ambient window bridge declarations for preload-exposed APIs.
 *
 * This file intentionally stays minimal. The actual runtime contract lives in
 * `src/workspace-contract.ts`; this declaration file only tells TypeScript what
 * globals the preload layer exposes on `window` inside renderer code.
 */

declare global {
  interface Window {
    launcher: LauncherApi
    workspace: WorkspaceApi
  }
}

export {}
