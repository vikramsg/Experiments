import path from 'node:path'

import { BaseWindow, WebContentsView } from 'electron'

import { TerminalAppearanceStore } from '../../features/terminal/main/TerminalAppearanceStore'
import { TerminalPtyService } from '../../features/terminal/main/TerminalPtyService'
import { IPC_CHANNELS } from '../../ipc'

export type TerminalBundle = {
  window: BaseWindow
  view: WebContentsView
  service: TerminalPtyService
  appearanceStore: TerminalAppearanceStore
}

function getTerminalEntryUrl() {
  if (TERMINAL_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL('src/app/renderer/entries/terminal.html', TERMINAL_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${TERMINAL_WINDOW_VITE_NAME}/src/app/renderer/entries/terminal.html`)
}

function loadTerminalPage(view: WebContentsView) {
  const target = getTerminalEntryUrl()

  if (target.startsWith('http')) {
    return view.webContents.loadURL(target)
  }

  return view.webContents.loadFile(target)
}

export async function createTerminalWindow(repoRoot: string, userDataPath: string): Promise<TerminalBundle> {
  const window = new BaseWindow({
    width: 1320,
    height: 900,
    title: 'Terminal',
    backgroundColor: '#11161a',
  })

  const view = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'terminal.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const appearanceStore = new TerminalAppearanceStore(userDataPath)
  const appearance = await appearanceStore.load()
  const service = new TerminalPtyService({ repoRoot, appearance })

  const applyBounds = () => {
    const bounds = window.getContentBounds()
    view.setBounds({
      x: 0,
      y: 0,
      width: bounds.width,
      height: bounds.height,
    })
  }

  const unsubscribeState = service.subscribe((state) => {
    view.webContents.send(IPC_CHANNELS.terminalState, state)
  })
  const unsubscribeData = service.onData((data) => {
    view.webContents.send(IPC_CHANNELS.terminalData, data)
  })

  window.contentView.addChildView(view)
  applyBounds()

  window.on('resize', applyBounds)
  window.on('closed', () => {
    unsubscribeState()
    unsubscribeData()
    void service.dispose()
    view.webContents.close()
  })

  view.webContents.once('did-finish-load', () => {
    view.webContents.send(IPC_CHANNELS.terminalState, service.getState())
  })

  await loadTerminalPage(view)

  return {
    window,
    view,
    service,
    appearanceStore,
  }
}
