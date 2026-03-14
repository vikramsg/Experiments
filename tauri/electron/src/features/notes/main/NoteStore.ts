import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import {
  DEFAULT_WORKSPACE_SNAPSHOT,
  toPersistedWorkspaceSnapshot,
  type PersistedWorkspaceSnapshot,
  type WorkspaceSnapshot,
} from '../../../workspace-model'

export class NoteStore {
  private readonly filePath: string

  constructor(userDataPath: string) {
    this.filePath = join(userDataPath, 'workspace-state.json')
  }

  async load(): Promise<WorkspaceSnapshot> {
    try {
      const raw = await readFile(this.filePath, 'utf8')
      const persisted = JSON.parse(raw) as Partial<PersistedWorkspaceSnapshot>

      return {
        ...DEFAULT_WORKSPACE_SNAPSHOT,
        ...persisted,
      }
    } catch {
      return DEFAULT_WORKSPACE_SNAPSHOT
    }
  }

  async save(snapshot: WorkspaceSnapshot): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
    // History availability is derived from the live browser webContents on each
    // launch, so only durable workspace fields are written to disk.
    await writeFile(this.filePath, JSON.stringify(toPersistedWorkspaceSnapshot(snapshot), null, 2), 'utf8')
  }
}
