import path from 'node:path'

import { app, BrowserWindow } from 'electron'
import started from 'electron-squirrel-startup'

import { createOpenCodeWindow, type OpenCodeBundle } from './create-opencode-window'
import { createLauncherWindow } from './create-launcher-window'
import { createTerminalWindow, type TerminalBundle } from './create-terminal-window'
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
let terminalBundle: TerminalBundle | null = null

function resolveRepoRoot() {
  if (process.env.ELECTRON_OPENCODE_REPO_ROOT) {
    return process.env.ELECTRON_OPENCODE_REPO_ROOT
  }

  if (process.env.ELECTRON_TERMINAL_REPO_ROOT) {
    return process.env.ELECTRON_TERMINAL_REPO_ROOT
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

  openCodeBundle = await createOpenCodeWindow(resolveRepoRoot())
  openCodeBundle.window.on('closed', () => {
    openCodeBundle = null
  })
}

async function openTerminal() {
  if (terminalBundle) {
    terminalBundle.window.show()
    return
  }

  terminalBundle = await createTerminalWindow(resolveRepoRoot(), app.getPath('userData'))
  terminalBundle.window.on('closed', () => {
    terminalBundle = null
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

function getTerminal() {
  return terminalBundle
}

function requireTerminal() {
  if (!terminalBundle) {
    throw new Error('Terminal is not open')
  }

  return terminalBundle
}

registerIpc({
  createWorkspace: openWorkspace,
  createOpenCode: openOpenCode,
  createTerminal: openTerminal,
  getWorkspace,
  getOpenCode,
  getTerminal,
  requireWorkspace,
  requireOpenCode,
  requireTerminal,
  openCodeRepoRoot: resolveRepoRoot(),
  terminalRepoRoot: resolveRepoRoot(),
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
