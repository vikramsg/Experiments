import path from 'node:path'

import { BaseWindow, WebContentsView } from 'electron'

import { createBrowserHost } from './browser-host'
import { NoteStore } from '../../features/notes/main/NoteStore'
import { WorkspaceController } from '../../features/workspace/main/WorkspaceController'
import { IPC_CHANNELS } from '../../ipc'

const BROWSER_PARTITION = 'persist:workspace-browser'

export type WorkspaceBundle = {
  window: BaseWindow
  notesView: WebContentsView
  splitterView: WebContentsView
  browserChromeView: WebContentsView
  browserView: WebContentsView
  browserHost: Awaited<ReturnType<typeof createBrowserHost>>
  controller: WorkspaceController
  store: NoteStore
}

function getWorkspaceAsset(page: 'notes.html' | 'splitter.html') {
  if (WORKSPACE_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL(`src/app/renderer/entries/${page}`, WORKSPACE_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${WORKSPACE_WINDOW_VITE_NAME}/src/app/renderer/entries/${page}`)
}

function loadLocalPage(view: WebContentsView, page: 'notes.html' | 'splitter.html') {
  const target = getWorkspaceAsset(page)

  if (target.startsWith('http')) {
    return view.webContents.loadURL(target)
  }

  return view.webContents.loadFile(target)
}

export async function createWorkspaceWindow(userDataPath: string): Promise<WorkspaceBundle> {
  const store = new NoteStore(userDataPath)
  const snapshot = await store.load()
  let controller: WorkspaceController | null = null

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

  const browserHost = await createBrowserHost({
    partition: BROWSER_PARTITION,
    initialUrl: snapshot.browserUrl,
    onStateChange: (browserState) => {
      if (!controller) {
        return
      }

      controller.setBrowserNavigationState(browserState)
      void store.save(controller.getSnapshot())
    },
  })
  const { browserChromeView, browserView } = browserHost

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
      },
    },
  }

  controller = new WorkspaceController(
    adapter,
    {
      notesView,
      splitterView,
      browserChromeView,
      browserView,
    },
    {
      ...snapshot,
      browserUrl: browserHost.getSnapshot().browserUrl,
      canGoBack: browserHost.getSnapshot().canGoBack,
      canGoForward: browserHost.getSnapshot().canGoForward,
    },
  )

  notesView.webContents.once('did-finish-load', () => {
    notesView.webContents.send(IPC_CHANNELS.workspaceState, controller.getSnapshot())
  })

  void Promise.all([
    loadLocalPage(notesView, 'notes.html'),
    loadLocalPage(splitterView, 'splitter.html'),
  ])

  if (!controller) {
    throw new Error('Workspace controller did not initialize')
  }

  const workspaceController = controller

  return {
    window,
    notesView,
    splitterView,
    browserChromeView,
    browserView,
    browserHost,
    controller: workspaceController,
    store,
  }
}
