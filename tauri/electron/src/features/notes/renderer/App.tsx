import { useEffect, useMemo, useState, type CSSProperties } from 'react'

import type { WorkspaceApi } from '../../../types'

export type NotesAppProps = {
  api: WorkspaceApi
}

export function App({ api }: NotesAppProps) {
  const [notes, setNotes] = useState('')
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    let active = true

    void api
      .loadState()
      .then((snapshot) => {
        if (!active) {
          return
        }

        setNotes(snapshot.notes)
        setIsLoaded(true)
      })
      .catch(() => {
        // Workspace state can arrive through the subscription path during startup.
      })

    const unsubscribe = api.onStateChange((snapshot) => {
      setNotes(snapshot.notes)
      setIsLoaded(true)
    })

    return () => {
      active = false
      unsubscribe()
    }
  }, [api])

  const status = useMemo(() => (isLoaded ? 'Auto-saving notes to your workspace.' : 'Loading workspace...'), [isLoaded])

  return (
    <main style={styles.page}>
      <header style={styles.header}>
        <div>
          <p style={styles.kicker}>Browser + Notes</p>
          <h1 style={styles.title}>Workspace Notes</h1>
        </div>
        <p style={styles.status}>{status}</p>
      </header>

      <label style={styles.label}>
        Notes Editor
        <textarea
          aria-label="Notes Editor"
          style={styles.editor}
          value={notes}
          onChange={(event) => {
            const nextValue = event.target.value
            setNotes(nextValue)
            void api.saveNotes(nextValue)
          }}
        />
      </label>
    </main>
  )
}

const styles: Record<string, CSSProperties> = {
  page: {
    minHeight: '100vh',
    boxSizing: 'border-box',
    padding: '20px',
    margin: 0,
    background: '#f8f4ea',
    color: '#2b2417',
    fontFamily: 'Georgia, serif',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '16px',
  },
  kicker: {
    margin: 0,
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    fontSize: '0.72rem',
    color: '#8a7452',
  },
  title: {
    margin: '6px 0 0',
    fontSize: '1.9rem',
  },
  status: {
    margin: 0,
    color: '#6a5a40',
    fontSize: '0.95rem',
  },
  label: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    fontWeight: 600,
    flex: 1,
  },
  editor: {
    flex: 1,
    minHeight: '60vh',
    resize: 'none',
    borderRadius: '16px',
    border: '1px solid #c7baa0',
    padding: '14px',
    font: 'inherit',
    lineHeight: 1.6,
    background: '#fffdf8',
  },
}
