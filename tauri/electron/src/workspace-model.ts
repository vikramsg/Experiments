/**
 * Canonical workspace data model for state that crosses process boundaries.
 *
 * This root file owns the durable workspace snapshot shape, the live browser
 * navigation fields, and helper functions that project live state down to the
 * persisted form written to disk. Keeping the model shallow at `src/` makes it
 * obvious that this is a stable application boundary rather than business logic
 * owned by a single feature or a miscellaneous `shared` dumping ground.
 *
 * Import guidance:
 * - `src/app/*` may import this file.
 * - `src/features/*` may import this file.
 * - features may not import each other; this file is the approved cross-cutting
 *   model boundary instead.
 */
export type PersistedWorkspaceSnapshot = {
  notes: string
  notesWidth: number
  browserUrl: string
}

export type WorkspaceSnapshot = PersistedWorkspaceSnapshot & {
  canGoBack: boolean
  canGoForward: boolean
}

export const DEFAULT_WORKSPACE_SNAPSHOT: WorkspaceSnapshot = {
  notes: '',
  notesWidth: 420,
  browserUrl: 'https://example.com',
  canGoBack: false,
  canGoForward: false,
}

export function toPersistedWorkspaceSnapshot(snapshot: WorkspaceSnapshot): PersistedWorkspaceSnapshot {
  return {
    notes: snapshot.notes,
    notesWidth: snapshot.notesWidth,
    browserUrl: snapshot.browserUrl,
  }
}
