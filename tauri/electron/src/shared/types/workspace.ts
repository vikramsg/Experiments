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
