import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type KeyboardEvent,
} from 'react'

import type { OpenCodeApi } from '../../../opencode-contract'
import { createDefaultOpenCodeState, type OpenCodeState } from '../../../opencode-model'

export type OpenCodeAppProps = {
  api: OpenCodeApi
}

export function App({ api }: OpenCodeAppProps) {
  const [state, setState] = useState<OpenCodeState>(() => createDefaultOpenCodeState('Loading repo scope...'))
  const [prompt, setPrompt] = useState('')
  const messageColumnRef = useRef<HTMLDivElement | null>(null)

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

  const isBusy = state.status === 'connecting' || state.status === 'responding'

  useEffect(() => {
    const messageColumn = messageColumnRef.current
    if (!messageColumn) {
      return
    }

    messageColumn.scrollTop = messageColumn.scrollHeight
  }, [state.messages])

  const submitPrompt = (event?: FormEvent) => {
    event?.preventDefault()
    const nextPrompt = prompt.trim()
    if (!nextPrompt || isBusy) {
      return
    }

    setPrompt('')
    void api.sendPrompt(nextPrompt).catch(() => {
      // Errors flow back through state updates from the main process service.
    })
  }

  const handlePromptKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) {
      return
    }

    event.preventDefault()
    submitPrompt()
  }

  return (
    <main style={styles.page}>
      <section style={styles.chatShell}>
        <div ref={messageColumnRef} style={styles.messageColumn}>
          {state.messages.map((message) => (
            <article key={message.id} style={message.role === 'assistant' ? styles.assistantMessage : message.role === 'user' ? styles.userMessage : styles.systemMessage}>
              <p style={styles.messageRole}>{message.role}</p>
              <p style={styles.messageText}>{message.text}</p>
            </article>
          ))}
        </div>

        <form style={styles.promptForm} onSubmit={submitPrompt}>
          <textarea
            aria-label="Ask OpenCode"
            style={styles.promptInput}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={handlePromptKeyDown}
            placeholder="Ask about architecture, files, behavior, what OpenCode sees in the browser, or where something lives in this repo."
          />
          <button style={styles.sendButton} type="submit" disabled={isBusy}>
            {isBusy ? 'Thinking...' : 'Send Prompt'}
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
    height: '100vh',
    boxSizing: 'border-box',
    margin: 0,
    padding: '18px',
    background:
      'radial-gradient(circle at top left, rgba(255, 226, 183, 0.55), transparent 32%), linear-gradient(180deg, #f7f0df 0%, #e7dcc5 100%)',
    color: '#2d261d',
    fontFamily: 'Georgia, serif',
    overflow: 'hidden',
  },
  chatShell: {
    height: '100%',
    minHeight: 0,
    display: 'grid',
    gridTemplateRows: 'minmax(0, 1fr) auto',
    gap: '14px',
    background: 'rgba(255, 251, 243, 0.58)',
    borderRadius: '24px',
    border: '1px solid rgba(75, 53, 26, 0.12)',
    boxShadow: '0 24px 64px rgba(69, 48, 23, 0.08)',
    padding: '16px',
    overflow: 'hidden',
  },
  messageColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
    overflowY: 'auto',
    minHeight: 0,
    paddingRight: '6px',
  },
  systemMessage: {
    ...messageCardBase,
    background: '#f7efd9',
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
    display: 'flex',
    gap: '12px',
    alignItems: 'flex-end',
  },
  promptInput: {
    flex: 1,
    minWidth: 0,
    minHeight: '96px',
    maxHeight: '180px',
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
    height: '52px',
    minWidth: '148px',
    padding: '0 22px',
    background: '#6c4a1e',
    color: '#fff9f0',
    fontSize: '1rem',
    fontWeight: 700,
    cursor: 'pointer',
    alignSelf: 'flex-end',
  },
}
