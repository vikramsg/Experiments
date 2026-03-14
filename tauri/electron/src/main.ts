import path from 'node:path'

import { app, BaseWindow, BrowserWindow, WebContentsView, ipcMain, session } from 'electron'
import started from 'electron-squirrel-startup'

import { NoteStore } from './main/note-store'
import { WorkspaceController } from './main/workspace-controller'
import { computeSplitLayout } from './shared/split-layout'

if (started) {
  app.quit()
}

if (process.env.ELECTRON_USER_DATA_DIR) {
  app.setPath('userData', process.env.ELECTRON_USER_DATA_DIR)
}

const SPLITTER_WIDTH = 12
const BROWSER_PARTITION = 'persist:workspace-browser'

type WorkspaceBundle = {
  window: BaseWindow
  notesView: WebContentsView
  splitterView: WebContentsView
  browserView: WebContentsView
  controller: WorkspaceController
  store: NoteStore
  syncLocalBounds: () => void
}

let launcherWindow: BrowserWindow | null = null
let workspaceBundle: WorkspaceBundle | null = null

function getLauncherEntryUrl() {
  if (LAUNCHER_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL('launcher.html', LAUNCHER_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${LAUNCHER_WINDOW_VITE_NAME}/launcher.html`)
}

function getWorkspaceAsset(page: 'notes.html' | 'splitter.html') {
  if (WORKSPACE_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL(page, WORKSPACE_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${WORKSPACE_WINDOW_VITE_NAME}/${page}`)
}

function loadLocalPage(view: WebContentsView, page: 'notes.html' | 'splitter.html') {
  const target = getWorkspaceAsset(page)

  if (target.startsWith('http')) {
    return view.webContents.loadURL(target)
  }

  return view.webContents.loadFile(target)
}

function createLauncherWindow() {
  launcherWindow = new BrowserWindow({
    width: 1100,
    height: 760,
    title: 'Electron Workspace',
    webPreferences: {
      preload: path.join(__dirname, 'launcher.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const target = getLauncherEntryUrl()
  if (target.startsWith('http')) {
    void launcherWindow.loadURL(target)
  } else {
    void launcherWindow.loadFile(target)
  }

  launcherWindow.on('closed', () => {
    launcherWindow = null
  })
}

function normalizeUrl(url: string) {
  if (/^https?:\/\//i.test(url)) {
    return url
  }

  return `https://${url}`
}

function configureBrowserSession() {
  const browserSession = session.fromPartition(BROWSER_PARTITION)
  browserSession.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false)
  })
  return browserSession
}

async function createWorkspace() {
  if (workspaceBundle) {
    workspaceBundle.window.show()
    return
  }

  const store = new NoteStore(app.getPath('userData'))
  const snapshot = await store.load()
  const browserSession = configureBrowserSession()

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

  const browserView = new WebContentsView({
    webPreferences: {
      session: browserSession,
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      navigateOnDragDrop: false,
    },
  })

  browserView.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))

  window.contentView.addChildView(notesView)
  window.contentView.addChildView(splitterView)
  window.contentView.addChildView(browserView)

  await Promise.all([loadLocalPage(notesView, 'notes.html'), loadLocalPage(splitterView, 'splitter.html')])
  await browserView.webContents.loadURL(normalizeUrl(snapshot.browserUrl))

  const adapter = {
    getContentBounds: () => window.getContentBounds(),
    on: window.on.bind(window),
    webContents: {
      send: (channel: string, payload: typeof snapshot) => {
        notesView.webContents.send(channel, payload)
        splitterView.webContents.send(channel, payload)
      },
    },
  }

  const controller = new WorkspaceController(adapter, browserView, {
    ...snapshot,
    browserUrl: normalizeUrl(snapshot.browserUrl),
  })

  const syncLocalBounds = () => {
    const bounds = window.getContentBounds()
    const layout = computeSplitLayout({
      windowWidth: bounds.width,
      windowHeight: bounds.height,
      notesWidth: controller.getSnapshot().notesWidth,
      splitterWidth: SPLITTER_WIDTH,
      minNotesWidth: 280,
      minBrowserWidth: 360,
    })

    notesView.setBounds({
      x: 0,
      y: 0,
      width: layout.notesWidth,
      height: bounds.height,
    })

    splitterView.setBounds({
      x: layout.splitterX,
      y: 0,
      width: SPLITTER_WIDTH,
      height: bounds.height,
    })
  }

  window.on('resize', syncLocalBounds)
  window.on('closed', () => {
    workspaceBundle = null
  })

  notesView.webContents.once('did-finish-load', () => {
    notesView.webContents.send('workspace:state', controller.getSnapshot())
  })

  splitterView.webContents.once('did-finish-load', () => {
    splitterView.webContents.send('workspace:state', controller.getSnapshot())
  })

  workspaceBundle = {
    window,
    notesView,
    splitterView,
    browserView,
    controller,
    store,
    syncLocalBounds,
  }

  syncLocalBounds()
}

function requireWorkspace() {
  if (!workspaceBundle) {
    throw new Error('Workspace is not open')
  }

  return workspaceBundle
}

ipcMain.handle('launcher:open-workspace', async () => {
  await createWorkspace()
})

ipcMain.handle('workspace:get-state', async () => {
  const workspace = requireWorkspace()
  return workspace.controller.getSnapshot()
})

ipcMain.handle('workspace:save-notes', async (_event, notes: string) => {
  const workspace = requireWorkspace()
  workspace.controller.setNotes(notes)
  await workspace.store.save(workspace.controller.getSnapshot())
})

ipcMain.handle('workspace:set-browser-url', async (_event, url: string) => {
  const workspace = requireWorkspace()
  const normalized = normalizeUrl(url)
  workspace.controller.setBrowserUrl(normalized)
  await workspace.browserView.webContents.loadURL(normalized)
  await workspace.store.save(workspace.controller.getSnapshot())
})

ipcMain.handle('workspace:adjust-splitter', async (_event, delta: number) => {
  const workspace = requireWorkspace()
  const current = workspace.controller.getSnapshot()
  workspace.controller.setNotesWidth(current.notesWidth + delta)
  workspace.syncLocalBounds()
  await workspace.store.save(workspace.controller.getSnapshot())
})

app.whenReady().then(() => {
  createLauncherWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createLauncherWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
