/**
 * Public renderer-facing contract for browser chrome surfaces.
 *
 * Browser UI should talk through this root contract instead of piggybacking on
 * workspace-only APIs. That keeps browser ownership separate from notes and
 * OpenCode while letting `app/*` compose multiple browser-backed windows.
 */
import type { BrowserSnapshot } from './browser-model'

export type BrowserApi = {
  loadState: () => Promise<BrowserSnapshot>
  setBrowserUrl: (url: string) => Promise<void>
  goBack: () => Promise<void>
  goForward: () => Promise<void>
  onStateChange: (listener: (snapshot: BrowserSnapshot) => void) => () => void
}
