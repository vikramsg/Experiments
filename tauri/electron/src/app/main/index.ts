import { app, BrowserWindow } from 'electron'
import started from 'electron-squirrel-startup'

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

function requireWorkspace() {
  if (!workspaceBundle) {
    throw new Error('Workspace is not open')
  }

  return workspaceBundle
}

registerIpc({
  createWorkspace: openWorkspace,
  requireWorkspace,
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
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
