/**
 * Ambient window bridge declarations for preload-exposed APIs.
 *
 * This file intentionally stays minimal. Runtime contracts live in the root
 * boundary files under `src/`; this declaration file only tells TypeScript what
 * globals the preload layer exposes on `window` inside renderer code.
 */
import type { LauncherApi, WorkspaceApi } from './workspace-contract'
import type { OpenCodeApi } from './opencode-contract'
import type { TerminalApi } from './terminal-contract'

declare global {
  interface Window {
    launcher: LauncherApi
    workspace: WorkspaceApi
    opencode: OpenCodeApi
    terminal: TerminalApi
  }
}

export {}
