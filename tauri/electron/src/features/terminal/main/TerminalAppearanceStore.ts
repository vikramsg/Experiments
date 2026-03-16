import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import type { Logger } from 'pino'

import { logger } from '../../../app/main/logger'
import {
  BUNDLED_TERMINAL_FONT_FAMILY,
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
    private readonly appLogger: Pick<Logger, 'warn'> = logger,
  ) {
    this.filePath = join(userDataPath, 'terminal-appearance.json')
  }

  async load(): Promise<TerminalAppearance> {
    const ghosttyFontFamily = await loadGhosttyConfigFontFamily(this.ghosttyConfigPath)
    this.logGhosttyFontMismatch(ghosttyFontFamily)

    try {
      const raw = await readFile(this.filePath, 'utf8')
      const persisted = JSON.parse(raw) as Partial<PersistedTerminalAppearance>
      const resolved = resolveTerminalAppearance({ saved: persisted })

      if (resolved.fontFamily !== persisted.fontFamily) {
        await this.save(resolved)
      }

      return resolved
    } catch {
      const resolved = resolveTerminalAppearance({})
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

  private logGhosttyFontMismatch(ghosttyFontFamily: string | null) {
    if (!ghosttyFontFamily || ghosttyFontFamily === BUNDLED_TERMINAL_FONT_FAMILY) {
      return
    }

    this.appLogger.warn(
      {
        ghosttyFontFamily,
        renderFontFamily: BUNDLED_TERMINAL_FONT_FAMILY,
      },
      'Ghostty terminal font differs from bundled Electron render font',
    )
  }
}
