import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

export function parseGhosttyConfigFontFamily(raw: string): string | null {
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) {
      continue
    }

    const match = trimmed.match(/^font-family\s*=\s*(.+)$/)
    if (!match) {
      continue
    }

    return match[1].trim().replace(/^['"]|['"]$/g, '') || null
  }

  return null
}

function resolveGhosttyConfigPath(explicitPath?: string): string {
  if (explicitPath) {
    return explicitPath
  }

  if (process.env.ELECTRON_GHOSTTY_CONFIG_PATH) {
    return process.env.ELECTRON_GHOSTTY_CONFIG_PATH
  }

  if (process.env.XDG_CONFIG_HOME) {
    return join(process.env.XDG_CONFIG_HOME, 'ghostty', 'config')
  }

  return join(homedir(), '.config', 'ghostty', 'config')
}

export async function loadGhosttyConfigFontFamily(explicitPath?: string): Promise<string | null> {
  try {
    // Current product assumption: Ghostty is the source of truth for the user's
    // preferred terminal text font, even though Electron currently renders the
    // terminal with one bundled patched mono font for consistent glyph coverage.
    const raw = await readFile(resolveGhosttyConfigPath(explicitPath), 'utf8')
    return parseGhosttyConfigFontFamily(raw)
  } catch {
    return null
  }
}
