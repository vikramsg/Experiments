import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'

import type { TerminalApi } from '../../../terminal-contract'
import { createDefaultTerminalState, type TerminalState } from '../../../terminal-model'
import { createGhosttyRuntime, type GhosttyRuntime } from './ghostty-runtime'

const DEFAULT_BOOT_COLS = 120
const DEFAULT_BOOT_ROWS = 32

export type TerminalAppProps = {
  api: TerminalApi
}

function describeStatus(state: TerminalState): string {
  if (state.error) {
    return state.error
  }

  switch (state.status) {
    case 'connecting':
      return 'Starting a local shell in the main process...'
    case 'ready':
      return 'Local shell is ready. Commands run with your user permissions.'
    case 'exited':
      return state.exitCode === null ? 'Shell exited.' : `Shell exited with code ${state.exitCode}.`
    case 'error':
      return 'The terminal hit an error.'
    default:
      return 'Preparing the terminal surface...'
  }
}

export function App({ api }: TerminalAppProps) {
  const [state, setState] = useState<TerminalState>(() => createDefaultTerminalState('Loading repo scope...', 'Loading shell...'))
  const [hasLoadedInitialState, setHasLoadedInitialState] = useState(false)
  const runtimeRef = useRef<GhosttyRuntime | null>(null)
  const pendingDataRef = useRef<string[]>([])
  const terminalContainerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let active = true

    void api
      .loadState()
      .then((nextState) => {
        if (!active) {
          return
        }

        setState(nextState)
        setHasLoadedInitialState(true)
      })
      .catch(() => {
        if (active) {
          setHasLoadedInitialState(true)
        }
      })

    const unsubscribeState = api.onStateChange((nextState) => {
      setState(nextState)
      setHasLoadedInitialState(true)
    })
    const unsubscribeData = api.onData((data) => {
      if (runtimeRef.current) {
        runtimeRef.current.write(data)
        return
      }

      pendingDataRef.current.push(data)
    })

    return () => {
      active = false
      unsubscribeState()
      unsubscribeData()
    }
  }, [api])

  useEffect(() => {
    const container = terminalContainerRef.current
    // Wait for the authoritative main-process state before creating the terminal
    // so the first renderer paint uses the persisted font preference instead of a
    // temporary fallback stack.
    if (!hasLoadedInitialState || !container || runtimeRef.current) {
      return
    }

    let active = true

    void createGhosttyRuntime({
      container,
      cols: DEFAULT_BOOT_COLS,
      rows: DEFAULT_BOOT_ROWS,
      appearance: {
        fontFamily: state.appearance.fontFamily,
        fontSize: state.appearance.fontSize,
        minimalChrome: state.appearance.minimalChrome,
      },
      onInput: (data) => {
        void api.write(data).catch(() => {
          // Main-process errors flow back through terminal state updates.
        })
      },
      onResize: (cols, rows) => {
        setState((current) => (current.cols === cols && current.rows === rows ? current : { ...current, cols, rows }))
        void api.resize(cols, rows).catch(() => {
          // Main-process errors flow back through terminal state updates.
        })
      },
    })
      .then(async (runtime) => {
        if (!active) {
          runtime.dispose()
          return
        }

        runtimeRef.current = runtime
        runtime.focus()

        for (const chunk of pendingDataRef.current) {
          runtime.write(chunk)
        }
        pendingDataRef.current = []

        const size = runtime.getSize()
        await api.connect(size.cols, size.rows).catch(() => {
          // Main-process errors flow back through terminal state updates.
        })
      })
      .catch(() => {
        // Initialization failures flow back through state if connect fails later.
      })

    return () => {
      active = false
      runtimeRef.current?.dispose()
      runtimeRef.current = null
    }
  }, [api, hasLoadedInitialState, state.appearance.fontFamily, state.appearance.fontSize, state.appearance.minimalChrome])

  const status = useMemo(() => describeStatus(state), [state])

  return (
    <main style={getPageStyles(state.appearance.minimalChrome)}>
      <div data-testid="terminal-status" role="status" aria-live="polite" style={styles.visuallyHidden}>
        {status}
      </div>

      <section style={getTerminalShellStyles(state.appearance.minimalChrome)}>
        <div ref={terminalContainerRef} style={styles.terminalViewport} aria-label="Terminal surface" />
        {state.error ? <div style={styles.errorOverlay}>{state.error}</div> : null}
      </section>
    </main>
  )
}

function getPageStyles(minimalChrome: boolean): CSSProperties {
  return {
    height: '100vh',
    boxSizing: 'border-box',
    margin: 0,
    padding: minimalChrome ? 0 : 12,
    background: '#101519',
    overflow: 'hidden',
  }
}

function getTerminalShellStyles(minimalChrome: boolean): CSSProperties {
  return {
    position: 'relative',
    width: '100%',
    height: '100%',
    padding: minimalChrome ? 0 : 12,
    boxSizing: 'border-box',
    background: '#101519',
  }
}

const styles: Record<string, CSSProperties> = {
  visuallyHidden: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    padding: 0,
    margin: '-1px',
    overflow: 'hidden',
    clip: 'rect(0, 0, 0, 0)',
    whiteSpace: 'nowrap',
    border: 0,
  },
  terminalViewport: {
    width: '100%',
    height: '100%',
    minHeight: 0,
    background: '#101519',
    overflow: 'hidden',
  },
  errorOverlay: {
    position: 'absolute',
    inset: '24px auto auto 24px',
    maxWidth: 'min(560px, calc(100% - 48px))',
    padding: '12px 14px',
    borderRadius: '10px',
    background: 'rgba(24, 12, 12, 0.92)',
    color: '#ffd6d6',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    fontSize: '0.95rem',
    lineHeight: 1.5,
    border: '1px solid rgba(255, 149, 149, 0.24)',
    boxShadow: '0 18px 32px rgba(0, 0, 0, 0.35)',
    pointerEvents: 'none',
  },
}
