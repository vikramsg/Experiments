import path from 'node:path'

import { BaseWindow, WebContentsView } from 'electron'

import type { BrowserContextSnapshot } from '../../features/browser/main/browser-context'
import { OpenCodeService } from '../../features/opencode/main/OpenCodeService'
import { IPC_CHANNELS } from '../../ipc'

export type OpenCodeBundle = {
  window: BaseWindow
  view: WebContentsView
  service: OpenCodeService
}

export type OpenCodeWindowOptions = {
  repoRoot: string
  browserMcp?: {
    url: string
    headers: Record<string, string>
  }
  browserContextProvider?: () => Promise<BrowserContextSnapshot | null>
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
    width: 1320,
    height: 900,
    title: 'OpenCode',
    backgroundColor: '#efe4cf',
  })

  const view = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'opencode.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const service = new OpenCodeService({
    repoRoot: input.repoRoot,
    browserMcp: input.browserMcp,
    browserContextProvider: input.browserContextProvider,
  })

  const applyBounds = () => {
    const bounds = window.getContentBounds()
    view.setBounds({
      x: 0,
      y: 0,
      width: bounds.width,
      height: bounds.height,
    })
  }

  const unsubscribe = service.subscribe((state) => {
    view.webContents.send(IPC_CHANNELS.opencodeState, state)
  })

  window.contentView.addChildView(view)
  applyBounds()

  window.on('resize', applyBounds)
  window.on('closed', () => {
    unsubscribe()
    void service.dispose()
    view.webContents.close()
  })

  view.webContents.once('did-finish-load', () => {
    view.webContents.send(IPC_CHANNELS.opencodeState, service.getState())
    void service.initialize().catch(() => {
      view.webContents.send(IPC_CHANNELS.opencodeState, service.getState())
    })
  })

  await loadOpenCodePage(view)

  return {
    window,
    view,
    service,
  }
}
