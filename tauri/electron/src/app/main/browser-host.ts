import path from 'node:path'

import { WebContentsView, session } from 'electron'

import type { BrowserSnapshot } from '../../browser-model'
import { DEFAULT_BROWSER_SNAPSHOT } from '../../browser-model'
import { IPC_CHANNELS } from '../../ipc'
import type { BrowserHistoryStore } from '../../features/browser/main/BrowserHistoryStore'
import {
  applyBrowserSecurityPolicy,
  normalizeUrl,
  readBrowserNavigationState,
  subscribeToBrowserNavigation,
} from '../../features/browser/main/browser-session'

export const BROWSER_CHROME_HEIGHT = 88

export type BrowserHost = {
  browserChromeView: WebContentsView
  browserView: WebContentsView
  getSnapshot: () => BrowserSnapshot
  publishState: () => void
  setBrowserUrl: (url: string) => Promise<void>
  goBack: () => void
  goForward: () => void
  close: () => void
  chromeSenderId: number
}

function getBrowserChromeAsset() {
  if (WORKSPACE_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL('src/app/renderer/entries/browser-chrome.html', WORKSPACE_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${WORKSPACE_WINDOW_VITE_NAME}/src/app/renderer/entries/browser-chrome.html`)
}

async function loadBrowserChromePage(view: WebContentsView) {
  const target = getBrowserChromeAsset()

  if (target.startsWith('http')) {
    await view.webContents.loadURL(target)
    return
  }

  await view.webContents.loadFile(target)
}

export async function createBrowserHost(input: {
  partition: string
  initialUrl: string
  historyStore: BrowserHistoryStore
  onStateChange?: (snapshot: BrowserSnapshot) => void
}): Promise<BrowserHost> {
  await input.historyStore.load()
  const browserSession = session.fromPartition(input.partition)

  const browserChromeView = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'browser.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const browserView = new WebContentsView({
    webPreferences: {
      session: browserSession,
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      navigateOnDragDrop: false,
    },
  })

  applyBrowserSecurityPolicy({
    session: browserSession,
    webContents: browserView.webContents,
  })

  let snapshot: BrowserSnapshot = {
    ...DEFAULT_BROWSER_SNAPSHOT,
    browserUrl: normalizeUrl(input.initialUrl),
    recentUrls: input.historyStore.getHistory(),
  }

  const updateHistory = (history: string[]) => {
    snapshot = {
      ...snapshot,
      recentUrls: history,
    }
    publishState()
  }

  const unsubscribeHistory = input.historyStore.subscribe(updateHistory)
  let closed = false

  const publishState = () => {
    browserChromeView.webContents.send(IPC_CHANNELS.browserState, snapshot)
    input.onStateChange?.(snapshot)
  }

  const updateStateFromWebContents = () => {
    snapshot = {
      ...snapshot,
      ...readBrowserNavigationState(browserView.webContents),
    }
    publishState()
  }

  subscribeToBrowserNavigation({
    webContents: browserView.webContents,
    onChange: (nextSnapshot) => {
      snapshot = {
        ...snapshot,
        ...nextSnapshot,
      }
      publishState()
      void input.historyStore.remember(nextSnapshot.browserUrl)
    },
  })

  browserChromeView.webContents.once('did-finish-load', () => {
    publishState()
  })

  await Promise.all([
    loadBrowserChromePage(browserChromeView),
    browserView.webContents.loadURL(snapshot.browserUrl),
  ])

  updateStateFromWebContents()
  await input.historyStore.remember(snapshot.browserUrl)

  return {
    browserChromeView,
    browserView,
    getSnapshot: () => snapshot,
    publishState,
    setBrowserUrl: async (url: string) => {
      const normalized = normalizeUrl(url)
      snapshot = {
        ...snapshot,
        browserUrl: normalized,
      }
      publishState()
      await browserView.webContents.loadURL(normalized)
      await input.historyStore.remember(normalized)
    },
    goBack: () => {
      if (browserView.webContents.canGoBack()) {
        browserView.webContents.goBack()
      }
    },
    goForward: () => {
      if (browserView.webContents.canGoForward()) {
        browserView.webContents.goForward()
      }
    },
    close: () => {
      if (closed) {
        return
      }

      closed = true
      unsubscribeHistory()
      browserChromeView.webContents.close()
      browserView.webContents.close()
    },
    chromeSenderId: browserChromeView.webContents.id,
  }
}
