import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

export type WorkspaceSnapshot = {
  notes: string
  notesWidth: number
  browserUrl: string
}

const DEFAULT_SNAPSHOT: WorkspaceSnapshot = {
  notes: '',
  notesWidth: 420,
  browserUrl: 'https://example.com',
}

export class NoteStore {
  private readonly filePath: string

  constructor(userDataPath: string) {
    this.filePath = join(userDataPath, 'workspace-state.json')
  }

  async load(): Promise<WorkspaceSnapshot> {
    try {
      const raw = await readFile(this.filePath, 'utf8')
      return {
        ...DEFAULT_SNAPSHOT,
        ...(JSON.parse(raw) as Partial<WorkspaceSnapshot>),
      }
    } catch {
      return DEFAULT_SNAPSHOT
    }
  }

  async save(snapshot: WorkspaceSnapshot): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
    await writeFile(this.filePath, JSON.stringify(snapshot, null, 2), 'utf8')
  }
}
