/**
 * Owns the renderer-facing data model for the Terminal app.
 *
 * The terminal feature, preload bridge, and main-process PTY service all depend
 * on this file to exchange stable state without introducing a generic shared
 * runtime bucket.
 */
export type TerminalAppearance = {
  fontFamily: string
  fontSize: number
  minimalChrome: boolean
}

export type PersistedTerminalAppearance = TerminalAppearance

export const BUNDLED_TERMINAL_FONT_FAMILY = 'Hack Nerd Font Mono'

export const DEFAULT_TERMINAL_APPEARANCE: TerminalAppearance = {
  fontFamily: BUNDLED_TERMINAL_FONT_FAMILY,
  fontSize: 14,
  minimalChrome: true,
}

export type TerminalStatus = 'idle' | 'connecting' | 'ready' | 'exited' | 'error'

export type TerminalState = {
  status: TerminalStatus
  cwd: string
  shell: string
  cols: number
  rows: number
  sessionId: string | null
  error: string | null
  exitCode: number | null
  appearance: TerminalAppearance
}

export function resolveTerminalAppearance(input: {
  saved?: Partial<TerminalAppearance> | null
}): TerminalAppearance {
  if (input.saved) {
    return {
      ...DEFAULT_TERMINAL_APPEARANCE,
      ...input.saved,
      fontFamily: BUNDLED_TERMINAL_FONT_FAMILY,
    }
  }

  return DEFAULT_TERMINAL_APPEARANCE
}

export function toGhosttyWebFontFamily(appearance: TerminalAppearance): string {
  // The Electron terminal renders with one bundled patched mono font so prompt
  // glyphs and cell metrics come from the same face instead of mixed fallbacks.
  return appearance.fontFamily
}

export function createDefaultTerminalState(
  repoRoot: string,
  shell: string,
  appearance: TerminalAppearance = DEFAULT_TERMINAL_APPEARANCE,
): TerminalState {
  return {
    status: 'idle',
    cwd: repoRoot,
    shell,
    cols: 120,
    rows: 32,
    sessionId: null,
    error: null,
    exitCode: null,
    appearance,
  }
}
