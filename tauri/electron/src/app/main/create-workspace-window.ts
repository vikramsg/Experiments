import path from 'node:path'

import { BaseWindow, WebContentsView, session } from 'electron'

import { applyBrowserSecurityPolicy, normalizeUrl } from '../../features/browser/main/browser-session'
import { NoteStore } from '../../features/notes/main/NoteStore'
import { WorkspaceController } from '../../features/workspace/main/WorkspaceController'
import { IPC_CHANNELS } from '../../shared/ipc/channels'

const BROWSER_PARTITION = 'persist:workspace-browser'

export type WorkspaceBundle = {
  window: BaseWindow
  notesView: WebContentsView
  splitterView: WebContentsView
  browserChromeView: WebContentsView
  browserView: WebContentsView
  controller: WorkspaceController
  store: NoteStore
}

function getWorkspaceAsset(page: 'notes.html' | 'splitter.html' | 'browser-chrome.html') {
  if (WORKSPACE_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL(`src/app/renderer/entries/${page}`, WORKSPACE_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${WORKSPACE_WINDOW_VITE_NAME}/src/app/renderer/entries/${page}`)
}

function loadLocalPage(view: WebContentsView, page: 'notes.html' | 'splitter.html' | 'browser-chrome.html') {
  const target = getWorkspaceAsset(page)

  if (target.startsWith('http')) {
    return view.webContents.loadURL(target)
  }

  return view.webContents.loadFile(target)
}

export async function createWorkspaceWindow(userDataPath: string): Promise<WorkspaceBundle> {
  const store = new NoteStore(userDataPath)
  const snapshot = await store.load()
  const browserSession = session.fromPartition(BROWSER_PARTITION)

  const window = new BaseWindow({
    width: 1400,
    height: 900,
    title: 'Browser + Notes',
    backgroundColor: '#e2ddd0',
  })

  const notesView = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'workspace.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const splitterView = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'workspace.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const browserChromeView = new WebContentsView({
    webPreferences: {
      preload: path.join(__dirname, 'workspace.js'),
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

  window.contentView.addChildView(notesView)
  window.contentView.addChildView(splitterView)
  window.contentView.addChildView(browserChromeView)
  window.contentView.addChildView(browserView)

  const adapter = {
    getContentBounds: () => window.getContentBounds(),
    on: window.on.bind(window),
    webContents: {
      send: (channel: string, payload: typeof snapshot) => {
        notesView.webContents.send(channel, payload)
        browserChromeView.webContents.send(channel, payload)
      },
    },
  }

  const controller = new WorkspaceController(
    adapter,
    {
      notesView,
      splitterView,
      browserChromeView,
      browserView,
    },
    {
      ...snapshot,
      browserUrl: normalizeUrl(snapshot.browserUrl),
    },
  )

  notesView.webContents.once('did-finish-load', () => {
    notesView.webContents.send(IPC_CHANNELS.workspaceState, controller.getSnapshot())
  })

  browserChromeView.webContents.once('did-finish-load', () => {
    browserChromeView.webContents.send(IPC_CHANNELS.workspaceState, controller.getSnapshot())
  })

  void Promise.all([
    loadLocalPage(notesView, 'notes.html'),
    loadLocalPage(splitterView, 'splitter.html'),
    loadLocalPage(browserChromeView, 'browser-chrome.html'),
  ])

  void browserView.webContents.loadURL(normalizeUrl(snapshot.browserUrl))

  return {
    window,
    notesView,
    splitterView,
    browserChromeView,
    browserView,
    controller,
    store,
  }
}
