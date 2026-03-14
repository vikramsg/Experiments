import type { CSSProperties } from 'react'

export type LauncherAppProps = {
  openWorkspace: () => Promise<void>
}

export function App({ openWorkspace }: LauncherAppProps) {
  return (
    <main style={styles.page}>
      <section style={styles.cardSection} aria-labelledby="launcher-title">
        <p style={styles.kicker}>Electron Workspace</p>
        <h1 id="launcher-title" style={styles.title}>
          Choose an App
        </h1>
        <article style={styles.card}>
          <p style={styles.eyebrow}>Ready now</p>
          <h2 style={styles.cardTitle}>Browser + Notes</h2>
          <p style={styles.body}>
            Open a split workspace with a note-taking editor on the left and a browser pane on the right.
          </p>
          <button style={styles.button} type="button" onClick={() => void openWorkspace()}>
            Launch Browser + Notes
          </button>
        </article>
      </section>
    </main>
  )
}

const styles: Record<string, CSSProperties> = {
  page: {
    minHeight: '100vh',
    margin: 0,
    padding: '48px',
    background: 'linear-gradient(160deg, #f6f4ee 0%, #e9ece6 100%)',
    color: '#1f2a21',
    fontFamily: 'Georgia, serif',
  },
  cardSection: {
    maxWidth: '720px',
  },
  kicker: {
    textTransform: 'uppercase',
    letterSpacing: '0.16em',
    fontSize: '0.78rem',
    margin: 0,
    color: '#5b6a5d',
  },
  title: {
    fontSize: '3rem',
    margin: '8px 0 24px',
  },
  card: {
    background: 'rgba(255, 255, 255, 0.72)',
    border: '1px solid rgba(31, 42, 33, 0.12)',
    borderRadius: '24px',
    padding: '24px',
    boxShadow: '0 18px 48px rgba(31, 42, 33, 0.08)',
  },
  eyebrow: {
    margin: 0,
    color: '#6d7d71',
    fontSize: '0.9rem',
  },
  cardTitle: {
    margin: '12px 0 8px',
    fontSize: '2rem',
  },
  body: {
    margin: '0 0 24px',
    lineHeight: 1.6,
  },
  button: {
    border: 'none',
    borderRadius: '999px',
    padding: '12px 20px',
    background: '#1f5c44',
    color: '#fffdf7',
    fontSize: '1rem',
    cursor: 'pointer',
  },
}
