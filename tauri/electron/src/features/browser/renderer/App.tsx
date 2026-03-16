import { useCallback, useEffect, useState, type CSSProperties } from 'react'

import type { BrowserApi } from '../../../browser-contract'
import type { BrowserSnapshot } from '../../../browser-model'

export type BrowserChromeAppProps = {
  api: BrowserApi
}

export function App({ api }: BrowserChromeAppProps) {
  const [browserUrl, setBrowserUrl] = useState('https://example.com')
  const [canGoBack, setCanGoBack] = useState(false)
  const [canGoForward, setCanGoForward] = useState(false)
  const [recentUrls, setRecentUrls] = useState<string[]>([])

  const applySnapshot = useCallback((snapshot: BrowserSnapshot) => {
    setBrowserUrl(snapshot.browserUrl)
    setCanGoBack(snapshot.canGoBack)
    setCanGoForward(snapshot.canGoForward)
    setRecentUrls(snapshot.recentUrls)
  }, [])

  useEffect(() => {
    let active = true

    void api
      .loadState()
      .then((snapshot) => {
        if (!active) {
          return
        }

        applySnapshot(snapshot)
      })
      .catch(() => {
        // Browser state can arrive through the subscription path during startup.
      })

    const unsubscribe = api.onStateChange((snapshot) => {
      applySnapshot(snapshot)
    })

    return () => {
      active = false
      unsubscribe()
    }
  }, [api, applySnapshot])

  const navigate = () => void api.setBrowserUrl(browserUrl)

  return (
    <main style={styles.page}>
      <div style={styles.row}>
        <button
          aria-label="Back"
          style={styles.navButton}
          type="button"
          onClick={() => void api.goBack()}
          disabled={!canGoBack}
        >
          {'←'}
        </button>
        <button
          aria-label="Forward"
          style={styles.navButton}
          type="button"
          onClick={() => void api.goForward()}
          disabled={!canGoForward}
        >
          {'→'}
        </button>
        <input
          aria-label="Browser URL"
          style={styles.input}
          value={browserUrl}
          list="browser-url-history"
          onChange={(event) => setBrowserUrl(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              navigate()
            }
          }}
        />
        <datalist id="browser-url-history">
          {recentUrls.map((url) => (
            <option key={url} value={url}>
              {url}
            </option>
          ))}
        </datalist>
        <button style={styles.button} type="button" onClick={navigate}>
          Go
        </button>
      </div>
    </main>
  )
}

const styles: Record<string, CSSProperties> = {
  page: {
    minHeight: '100vh',
    boxSizing: 'border-box',
    margin: 0,
    padding: '10px 16px',
    background: 'linear-gradient(180deg, #f2ecdf 0%, #e6ddca 100%)',
    color: '#2c2418',
    fontFamily: 'Georgia, serif',
    borderBottom: '1px solid rgba(88, 69, 43, 0.2)',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    minHeight: '68px',
  },
  input: {
    flex: 1,
    minWidth: 0,
    boxSizing: 'border-box',
    padding: '9px 12px',
    borderRadius: '999px',
    border: '1px solid #cdbd9f',
    background: '#fffbf3',
    fontSize: '0.98rem',
    color: '#2c2418',
  },
  navButton: {
    border: '1px solid #cdbd9f',
    borderRadius: '999px',
    padding: '10px 14px',
    background: '#fff8ef',
    color: '#5f4628',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: 700,
    flex: '0 0 auto',
    whiteSpace: 'nowrap',
    minWidth: '56px',
  },
  button: {
    border: 'none',
    borderRadius: '999px',
    padding: '10px 18px',
    background: '#7a4b19',
    color: '#fff8ef',
    cursor: 'pointer',
    fontSize: '0.98rem',
    flex: '0 0 auto',
    whiteSpace: 'nowrap',
  },
}
