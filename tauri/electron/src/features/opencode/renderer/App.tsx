import { useEffect, useMemo, useState, type CSSProperties, type FormEvent } from 'react'

import type { OpenCodeApi } from '../../../opencode-contract'
import { createDefaultOpenCodeState, type OpenCodeState } from '../../../opencode-model'

export type OpenCodeAppProps = {
  api: OpenCodeApi
}

function describeStatus(state: OpenCodeState): string {
  if (state.error) {
    return state.error
  }

  switch (state.status) {
    case 'connecting':
      return 'Connecting to the local OpenCode server...'
    case 'responding':
      return 'OpenCode is reading the repo and drafting a reply...'
    case 'ready':
      return 'Read-only repo chat is ready.'
    case 'error':
      return 'OpenCode hit an error.'
    default:
      return 'Starting OpenCode...'
  }
}

export function App({ api }: OpenCodeAppProps) {
  const [state, setState] = useState<OpenCodeState>(() => createDefaultOpenCodeState('Loading repo scope...'))
  const [prompt, setPrompt] = useState('')

  useEffect(() => {
    let active = true

    void api
      .loadState()
      .then((nextState) => {
        if (active) {
          setState(nextState)
        }
      })
      .catch(() => {
        // State updates can still arrive through the subscription path.
      })

    const unsubscribe = api.onStateChange((nextState) => {
      setState(nextState)
    })

    return () => {
      active = false
      unsubscribe()
    }
  }, [api])

  const status = useMemo(() => describeStatus(state), [state])

  const submitPrompt = (event: FormEvent) => {
    event.preventDefault()
    const nextPrompt = prompt.trim()
    if (!nextPrompt) {
      return
    }

    setPrompt('')
    void api.sendPrompt(nextPrompt).catch(() => {
      // Errors flow back through state updates from the main process service.
    })
  }

  return (
    <main style={styles.page}>
      <section style={styles.hero}>
        <div>
          <p style={styles.kicker}>OpenCode</p>
          <h1 style={styles.title}>OpenCode</h1>
          <p style={styles.lede}>Read-only repo chat with a local OpenCode server behind a narrow Electron bridge.</p>
        </div>
        <div style={styles.scopeCard}>
          <p style={styles.scopeLabel}>Repo scope</p>
          <p style={styles.scopeValue}>{state.repoRoot}</p>
          <p style={styles.status}>{status}</p>
        </div>
      </section>

      <section style={styles.chatShell}>
        <div style={styles.messageColumn}>
          {state.messages.map((message) => (
            <article key={message.id} style={message.role === 'assistant' ? styles.assistantMessage : message.role === 'user' ? styles.userMessage : styles.systemMessage}>
              <p style={styles.messageRole}>{message.role}</p>
              <p style={styles.messageText}>{message.text}</p>
            </article>
          ))}
        </div>

        <form style={styles.promptForm} onSubmit={submitPrompt}>
          <label style={styles.promptLabel}>
            Ask OpenCode
            <textarea
              aria-label="Ask OpenCode"
              style={styles.promptInput}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Ask about architecture, files, behavior, or where something lives in this repo."
            />
          </label>
          <button style={styles.sendButton} type="submit" disabled={state.status === 'connecting' || state.status === 'responding'}>
            Send Prompt
          </button>
        </form>
      </section>
    </main>
  )
}

const messageCardBase: CSSProperties = {
  borderRadius: '22px',
  padding: '16px 18px',
  display: 'flex',
  flexDirection: 'column',
  gap: '8px',
}

const styles: Record<string, CSSProperties> = {
  page: {
    minHeight: '100vh',
    boxSizing: 'border-box',
    margin: 0,
    padding: '24px',
    background:
      'radial-gradient(circle at top left, rgba(255, 226, 183, 0.55), transparent 32%), linear-gradient(180deg, #f7f0df 0%, #e7dcc5 100%)',
    color: '#2d261d',
    fontFamily: 'Georgia, serif',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  hero: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1.3fr) minmax(260px, 0.9fr)',
    gap: '18px',
    alignItems: 'stretch',
  },
  kicker: {
    margin: 0,
    textTransform: 'uppercase',
    letterSpacing: '0.16em',
    color: '#8b6f43',
    fontSize: '0.78rem',
  },
  title: {
    margin: '8px 0 10px',
    fontSize: '2.75rem',
  },
  lede: {
    margin: 0,
    lineHeight: 1.65,
    maxWidth: '52ch',
  },
  scopeCard: {
    borderRadius: '24px',
    padding: '18px 20px',
    background: 'rgba(255, 251, 242, 0.78)',
    border: '1px solid rgba(75, 53, 26, 0.12)',
    boxShadow: '0 18px 48px rgba(69, 48, 23, 0.08)',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  scopeLabel: {
    margin: 0,
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    color: '#8b6f43',
    fontSize: '0.74rem',
  },
  scopeValue: {
    margin: 0,
    fontSize: '1rem',
    lineHeight: 1.55,
    wordBreak: 'break-word',
  },
  status: {
    margin: 0,
    color: '#5f4b30',
    lineHeight: 1.55,
  },
  chatShell: {
    flex: 1,
    minHeight: 0,
    display: 'grid',
    gridTemplateRows: 'minmax(0, 1fr) auto',
    gap: '18px',
    background: 'rgba(255, 251, 243, 0.72)',
    borderRadius: '28px',
    border: '1px solid rgba(75, 53, 26, 0.12)',
    boxShadow: '0 24px 64px rgba(69, 48, 23, 0.08)',
    padding: '20px',
  },
  messageColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
    overflowY: 'auto',
    minHeight: 0,
  },
  systemMessage: {
    ...messageCardBase,
    background: '#f8f0db',
    border: '1px dashed rgba(111, 82, 40, 0.22)',
    alignSelf: 'stretch',
  },
  userMessage: {
    ...messageCardBase,
    background: '#fff7ea',
    border: '1px solid rgba(111, 82, 40, 0.14)',
    alignSelf: 'flex-end',
    maxWidth: '82%',
  },
  assistantMessage: {
    ...messageCardBase,
    background: '#efe4ce',
    border: '1px solid rgba(111, 82, 40, 0.14)',
    alignSelf: 'flex-start',
    maxWidth: '88%',
  },
  messageRole: {
    margin: 0,
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    fontSize: '0.72rem',
    color: '#86673a',
  },
  messageText: {
    margin: 0,
    lineHeight: 1.65,
    whiteSpace: 'pre-wrap',
  },
  promptForm: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    gap: '16px',
    alignItems: 'end',
  },
  promptLabel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    fontWeight: 600,
  },
  promptInput: {
    width: '100%',
    minHeight: '108px',
    boxSizing: 'border-box',
    resize: 'vertical',
    borderRadius: '20px',
    border: '1px solid #cdb488',
    background: '#fffdf7',
    padding: '14px 16px',
    font: 'inherit',
    color: '#2d261d',
  },
  sendButton: {
    border: 'none',
    borderRadius: '999px',
    padding: '14px 22px',
    background: '#6c4a1e',
    color: '#fff9f0',
    fontSize: '1rem',
    cursor: 'pointer',
    alignSelf: 'stretch',
  },
}
