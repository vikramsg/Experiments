/**
 * Owns the renderer-facing data model for the OpenCode app.
 *
 * The OpenCode feature, preload bridge, and main-process service can all depend
 * on this file to exchange chat state. Keeping the model in a shallow root file
 * makes the cross-cutting boundary explicit without growing `src/shared/` into
 * a generic runtime bucket.
 */
export type OpenCodeMessage = {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
}

export type OpenCodeStatus = 'idle' | 'connecting' | 'ready' | 'responding' | 'error'

export type BrowserToolStatus = 'checking' | 'ready' | 'unavailable'

export type OpenCodeState = {
  status: OpenCodeStatus
  repoRoot: string
  sessionId: string | null
  messages: OpenCodeMessage[]
  error: string | null
  browserToolStatus: BrowserToolStatus
  browserToolMessage: string | null
}

export function createDefaultOpenCodeState(repoRoot: string): OpenCodeState {
  return {
    status: 'idle',
    repoRoot,
    sessionId: null,
    messages: [
      {
        id: 'system-welcome',
        role: 'system',
        text:
          'Read-only repo chat is ready. Ask about files, architecture, behavior in this repo, or ask what OpenCode sees in the browser on the right.',
      },
    ],
    error: null,
    browserToolStatus: 'checking',
    browserToolMessage: null,
  }
}
