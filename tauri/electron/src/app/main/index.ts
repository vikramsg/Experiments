import path from 'node:path'

import { app, BrowserWindow } from 'electron'
import started from 'electron-squirrel-startup'

import { getBrowserContextSnapshot } from '../../features/browser/main/browser-context'
import { BrowserMcpServer } from '../../features/opencode/main/BrowserMcpServer'
import { createOpenCodeWindow, type OpenCodeBundle } from './create-opencode-window'
import { createLauncherWindow } from './create-launcher-window'
import { createWorkspaceWindow, type WorkspaceBundle } from './create-workspace-window'
import { registerIpc } from './register-ipc'

if (started) {
  app.quit()
}

if (process.env.ELECTRON_USER_DATA_DIR) {
  app.setPath('userData', process.env.ELECTRON_USER_DATA_DIR)
}

let launcherWindow: BrowserWindow | null = null
let workspaceBundle: WorkspaceBundle | null = null
let openCodeBundle: OpenCodeBundle | null = null
let browserMcpServer: BrowserMcpServer | null = null

function resolveOpenCodeRepoRoot() {
  if (process.env.ELECTRON_OPENCODE_REPO_ROOT) {
    return process.env.ELECTRON_OPENCODE_REPO_ROOT
  }

  return path.resolve(__dirname, '../../..')
}

async function openWorkspace() {
  if (workspaceBundle) {
    workspaceBundle.window.show()
    return
  }

  workspaceBundle = await createWorkspaceWindow(app.getPath('userData'))
  workspaceBundle.window.on('closed', () => {
    workspaceBundle = null
  })
}

async function openOpenCode() {
  if (openCodeBundle) {
    openCodeBundle.window.show()
    return
  }

  browserMcpServer ??= new BrowserMcpServer({
    getBrowserContext: async () => {
      const workspace = getWorkspace()
      if (!workspace) {
        return null
      }

      return await getBrowserContextSnapshot(workspace.browserView.webContents)
    },
  })

  const browserMcp = await browserMcpServer.start()

  openCodeBundle = await createOpenCodeWindow({
    repoRoot: resolveOpenCodeRepoRoot(),
    browserMcp,
    browserContextProvider: async () => {
      const workspace = getWorkspace()
      if (!workspace) {
        return null
      }

      return await getBrowserContextSnapshot(workspace.browserView.webContents)
    },
  })
  openCodeBundle.window.on('closed', () => {
    void browserMcpServer?.stop()
    browserMcpServer = null
    openCodeBundle = null
  })
}

function getWorkspace() {
  return workspaceBundle
}

function requireWorkspace() {
  if (!workspaceBundle) {
    throw new Error('Workspace is not open')
  }

  return workspaceBundle
}

function getOpenCode() {
  return openCodeBundle
}

function requireOpenCode() {
  if (!openCodeBundle) {
    throw new Error('OpenCode is not open')
  }

  return openCodeBundle
}

registerIpc({
  createWorkspace: openWorkspace,
  createOpenCode: openOpenCode,
  getWorkspace,
  getOpenCode,
  requireWorkspace,
  requireOpenCode,
  openCodeRepoRoot: resolveOpenCodeRepoRoot(),
})

app.whenReady().then(() => {
  launcherWindow = createLauncherWindow()
  launcherWindow.on('closed', () => {
    launcherWindow = null
  })

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      launcherWindow = createLauncherWindow()
      launcherWindow.on('closed', () => {
        launcherWindow = null
      })
    }
  })
})

app.on('window-all-closed', () => {
  void browserMcpServer?.stop()
  browserMcpServer = null

  if (process.platform !== 'darwin') {
    app.quit()
  }
})
