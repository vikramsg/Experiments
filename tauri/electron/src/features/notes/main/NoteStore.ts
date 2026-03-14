import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { DEFAULT_WORKSPACE_SNAPSHOT, type WorkspaceSnapshot } from '../../../shared/types/workspace'

export class NoteStore {
  private readonly filePath: string

  constructor(userDataPath: string) {
    this.filePath = join(userDataPath, 'workspace-state.json')
  }

  async load(): Promise<WorkspaceSnapshot> {
    try {
      const raw = await readFile(this.filePath, 'utf8')
      return {
        ...DEFAULT_WORKSPACE_SNAPSHOT,
        ...(JSON.parse(raw) as Partial<WorkspaceSnapshot>),
      }
    } catch {
      return DEFAULT_WORKSPACE_SNAPSHOT
    }
  }

  async save(snapshot: WorkspaceSnapshot): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
    await writeFile(this.filePath, JSON.stringify(snapshot, null, 2), 'utf8')
  }
}
