import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import {
  DEFAULT_TERMINAL_APPEARANCE,
  resolveTerminalAppearance,
  type PersistedTerminalAppearance,
  type TerminalAppearance,
} from '../../../terminal-model'

import { loadGhosttyConfigFontFamily } from './ghostty-config'

export class TerminalAppearanceStore {
  private readonly filePath: string

  constructor(
    userDataPath: string,
    private readonly ghosttyConfigPath?: string,
  ) {
    this.filePath = join(userDataPath, 'terminal-appearance.json')
  }

  async load(): Promise<TerminalAppearance> {
    try {
      const raw = await readFile(this.filePath, 'utf8')
      const persisted = JSON.parse(raw) as Partial<PersistedTerminalAppearance>

      return resolveTerminalAppearance({ saved: persisted })
    } catch {
      // Seed the Electron-owned terminal preferences once from Ghostty so the
      // first renderer paint matches the user's existing terminal more closely.
      const importedFontFamily = await loadGhosttyConfigFontFamily(this.ghosttyConfigPath)
      const resolved = resolveTerminalAppearance({ importedFontFamily })
      await this.save(resolved)
      return resolved
    }
  }

  async save(appearance: TerminalAppearance): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
    await writeFile(
      this.filePath,
      JSON.stringify(
        {
          ...DEFAULT_TERMINAL_APPEARANCE,
          ...appearance,
        } satisfies PersistedTerminalAppearance,
        null,
        2,
      ),
      'utf8',
    )
  }
}
