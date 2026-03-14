/**
 * Owns the public OpenCode renderer contract exposed through `window.opencode`.
 *
 * The renderer imports these types, preload implements the bridge, and the main
 * process stays authoritative over the server lifecycle and repo permissions.
 * Housing this contract at the root keeps the cross-cutting API shallow and
 * avoids turning `src/shared/` into a dumping ground.
 */
import type { OpenCodeState } from './opencode-model'

export type OpenCodeApi = {
  loadState: () => Promise<OpenCodeState>
  sendPrompt: (prompt: string) => Promise<void>
  onStateChange: (listener: (state: OpenCodeState) => void) => () => void
}
