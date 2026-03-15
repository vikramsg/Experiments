import path from 'node:path'

import { BaseWindow, WebContentsView } from 'electron'

import { getBrowserContextSnapshot } from '../../features/browser/main/browser-context'
import { BrowserMcpServer } from '../../features/browser/main/BrowserMcpServer'
import { OpenCodeService } from '../../features/opencode/main/OpenCodeService'
import { IPC_CHANNELS } from '../../ipc'
import { BROWSER_CHROME_HEIGHT, createBrowserHost, type BrowserHost } from './browser-host'

const OPEN_CODE_LEFT_WIDTH = 520
const MIN_OPEN_CODE_WIDTH = 420
const MIN_BROWSER_WIDTH = 420
const BROWSER_PARTITION = 'persist:opencode-browser'

export type OpenCodeBundle = {
  window: BaseWindow
  openCodeView: WebContentsView
  browserChromeView: WebContentsView
  browserView: WebContentsView
  browserHost: BrowserHost
  service: OpenCodeService
}

export type OpenCodeWindowOptions = {
  repoRoot: string
}

function clampOpenCodeWidth(windowWidth: number) {
  const maxLeftWidth = Math.max(MIN_OPEN_CODE_WIDTH, windowWidth - MIN_BROWSER_WIDTH)
  return Math.min(Math.max(OPEN_CODE_LEFT_WIDTH, MIN_OPEN_CODE_WIDTH), maxLeftWidth)
}

function getOpenCodeEntryUrl() {
  if (OPENCODE_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL('src/app/renderer/entries/opencode.html', OPENCODE_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${OPENCODE_WINDOW_VITE_NAME}/src/app/renderer/entries/opencode.html`)
}

function loadOpenCodePage(view: WebContentsView) {
  const target = getOpenCodeEntryUrl()

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

  const browserHost = await createBrowserHost({
    partition: BROWSER_PARTITION,
    initialUrl: 'https://example.com',
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

  const applyBounds = () => {
    const bounds = window.getContentBounds()
    const openCodeWidth = clampOpenCodeWidth(bounds.width)
    const rightPaneX = openCodeWidth
    const rightPaneWidth = Math.max(0, bounds.width - openCodeWidth)
    const chromeHeight = Math.min(BROWSER_CHROME_HEIGHT, bounds.height)

    openCodeView.setBounds({
      x: 0,
      y: 0,
      width: openCodeWidth,
      height: bounds.height,
    })

    browserHost.browserChromeView.setBounds({
      x: rightPaneX,
      y: 0,
      width: rightPaneWidth,
      height: chromeHeight,
    })

    browserHost.browserView.setBounds({
      x: rightPaneX,
      y: chromeHeight,
      width: rightPaneWidth,
      height: Math.max(0, bounds.height - chromeHeight),
    })
  }

  const unsubscribe = service.subscribe((state) => {
    openCodeView.webContents.send(IPC_CHANNELS.opencodeState, state)
  })

  window.contentView.addChildView(openCodeView)
  window.contentView.addChildView(browserHost.browserChromeView)
  window.contentView.addChildView(browserHost.browserView)
  applyBounds()

  window.on('resize', applyBounds)
  window.on('closed', () => {
    unsubscribe()
    void browserMcpServer.stop()
    void service.dispose()
    browserHost.close()
    openCodeView.webContents.close()
  })

  openCodeView.webContents.once('did-finish-load', () => {
    openCodeView.webContents.send(IPC_CHANNELS.opencodeState, service.getState())
    void service.initialize().catch(() => {
      openCodeView.webContents.send(IPC_CHANNELS.opencodeState, service.getState())
    })
  })

  await loadOpenCodePage(openCodeView)

  return {
    window,
    openCodeView,
    browserChromeView: browserHost.browserChromeView,
    browserView: browserHost.browserView,
    browserHost,
    service,
  }
}
