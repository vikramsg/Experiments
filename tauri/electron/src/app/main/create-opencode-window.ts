import path from 'node:path'

import { BaseWindow, WebContentsView } from 'electron'

import type { BrowserHistoryStore } from '../../features/browser/main/BrowserHistoryStore'
import { getBrowserContextSnapshot } from '../../features/browser/main/browser-context'
import { BrowserMcpServer } from '../../features/browser/main/BrowserMcpServer'
import { OpenCodeService } from '../../features/opencode/main/OpenCodeService'
import { IPC_CHANNELS } from '../../ipc'
import { OpenCodeBrowserController } from './OpenCodeBrowserController'
import { createBrowserHost, type BrowserHost } from './browser-host'

const OPEN_CODE_LEFT_WIDTH = 520
const BROWSER_PARTITION = 'persist:opencode-browser'

export type OpenCodeBundle = {
  window: BaseWindow
  openCodeView: WebContentsView
  splitterView: WebContentsView
  browserChromeView: WebContentsView
  browserView: WebContentsView
  browserHost: BrowserHost
  controller: OpenCodeBrowserController
  service: OpenCodeService
}

export type OpenCodeWindowOptions = {
  repoRoot: string
  browserHistoryStore: BrowserHistoryStore
}

function getOpenCodeEntryUrl() {
  if (OPENCODE_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL('src/app/renderer/entries/opencode.html', OPENCODE_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${OPENCODE_WINDOW_VITE_NAME}/src/app/renderer/entries/opencode.html`)
}

function getOpenCodeSplitterEntryUrl() {
  if (OPENCODE_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL('src/app/renderer/entries/opencode-splitter.html', OPENCODE_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${OPENCODE_WINDOW_VITE_NAME}/src/app/renderer/entries/opencode-splitter.html`)
}

function loadOpenCodePage(view: WebContentsView) {
  const target = getOpenCodeEntryUrl()

  if (target.startsWith('http')) {
    return view.webContents.loadURL(target)
  }

  return view.webContents.loadFile(target)
}

function loadOpenCodeSplitterPage(view: WebContentsView) {
  const target = getOpenCodeSplitterEntryUrl()

  if (target.startsWith('http')) {
    return view.webContents.loadURL(target)
  }

  return view.webContents.loadFile(target)
}

export async function createOpenCodeWindow(input: OpenCodeWindowOptions): Promise<OpenCodeBundle> {
  const window = new BaseWindow({
    width: 1440,
    height: 920,
    title: 'OpenCode + Browser',
    backgroundColor: '#efe4cf',
  })

  const openCodeView = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'opencode.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const splitterView = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'opencode.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const browserHost = await createBrowserHost({
    partition: BROWSER_PARTITION,
    initialUrl: 'https://example.com',
    historyStore: input.browserHistoryStore,
  })

  const browserMcpServer = new BrowserMcpServer({
    getBrowserContext: async () => await getBrowserContextSnapshot(browserHost.browserView.webContents),
  })
  const browserMcp = await browserMcpServer.start()

  const service = new OpenCodeService({
    repoRoot: input.repoRoot,
    browserMcp,
    browserContextProvider: async () => await getBrowserContextSnapshot(browserHost.browserView.webContents),
    getBrowserToolCallCount: () => browserMcpServer.getToolCallCount(),
  })

  const unsubscribe = service.subscribe((state) => {
    openCodeView.webContents.send(IPC_CHANNELS.opencodeState, state)
  })

  window.contentView.addChildView(openCodeView)
  window.contentView.addChildView(splitterView)
  window.contentView.addChildView(browserHost.browserChromeView)
  window.contentView.addChildView(browserHost.browserView)

  const controller = new OpenCodeBrowserController(
    window,
    {
      openCodeView,
      splitterView,
      browserChromeView: browserHost.browserChromeView,
      browserView: browserHost.browserView,
    },
    OPEN_CODE_LEFT_WIDTH,
  )

  window.on('closed', () => {
    unsubscribe()
    void browserMcpServer.stop()
    void service.dispose()
    browserHost.close()
  })

  openCodeView.webContents.once('did-finish-load', () => {
    openCodeView.webContents.send(IPC_CHANNELS.opencodeState, service.getState())
    void service.initialize().catch(() => {
      openCodeView.webContents.send(IPC_CHANNELS.opencodeState, service.getState())
    })
  })

  await Promise.all([loadOpenCodePage(openCodeView), loadOpenCodeSplitterPage(splitterView)])

  return {
    window,
    openCodeView,
    splitterView,
    browserChromeView: browserHost.browserChromeView,
    browserView: browserHost.browserView,
    browserHost,
    controller,
    service,
  }
}
