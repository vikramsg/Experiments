/**
 * Owns the public Terminal renderer contract exposed through `window.terminal`.
 *
 * The renderer imports these types, preload implements the bridge, and the main
 * process stays authoritative over the PTY lifecycle and shell permissions.
 */
import type { TerminalState } from './terminal-model'

export type TerminalApi = {
  loadState: () => Promise<TerminalState>
  connect: (cols: number, rows: number) => Promise<void>
  write: (data: string) => Promise<void>
  resize: (cols: number, rows: number) => Promise<void>
  restart: (cols: number, rows: number) => Promise<void>
  onData: (listener: (data: string) => void) => () => void
  onStateChange: (listener: (state: TerminalState) => void) => () => void
}
