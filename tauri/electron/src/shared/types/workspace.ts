export type WorkspaceSnapshot = {
  notes: string
  notesWidth: number
  browserUrl: string
}

export const DEFAULT_WORKSPACE_SNAPSHOT: WorkspaceSnapshot = {
  notes: '',
  notesWidth: 420,
  browserUrl: 'https://example.com',
}
