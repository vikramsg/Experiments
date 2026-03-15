import { ipcMain } from 'electron'

import { normalizeUrl } from '../../features/browser/main/browser-session'
import { resolveTerminalShell } from '../../features/terminal/main/TerminalPtyService'
import { createDefaultOpenCodeState } from '../../opencode-model'
import { IPC_CHANNELS } from '../../ipc'
import { createDefaultTerminalState } from '../../terminal-model'
import { DEFAULT_WORKSPACE_SNAPSHOT } from '../../workspace-model'
import type { OpenCodeBundle } from './create-opencode-window'
import type { TerminalBundle } from './create-terminal-window'
import type { WorkspaceBundle } from './create-workspace-window'

export function registerIpc(input: {
  createWorkspace: () => Promise<void>
  createOpenCode: () => Promise<void>
  createTerminal: () => Promise<void>
  getWorkspace: () => WorkspaceBundle | null
  getOpenCode: () => OpenCodeBundle | null
  getTerminal: () => TerminalBundle | null
  requireWorkspace: () => WorkspaceBundle
  requireOpenCode: () => OpenCodeBundle
  requireTerminal: () => TerminalBundle
  openCodeRepoRoot: string
  terminalRepoRoot: string
}) {
  ipcMain.handle(IPC_CHANNELS.launcherOpenWorkspace, async () => {
    await input.createWorkspace()
  })

  ipcMain.handle(IPC_CHANNELS.launcherOpenOpenCode, async () => {
    await input.createOpenCode()
  })

  ipcMain.handle(IPC_CHANNELS.launcherOpenTerminal, async () => {
    await input.createTerminal()
  })

  ipcMain.handle(IPC_CHANNELS.workspaceGetState, async () => {
    const workspace = input.getWorkspace()
    return workspace ? workspace.controller.getSnapshot() : DEFAULT_WORKSPACE_SNAPSHOT
  })

  ipcMain.handle(IPC_CHANNELS.workspaceSaveNotes, async (_event, notes: string) => {
    const workspace = input.requireWorkspace()
    workspace.controller.setNotes(notes)
    await workspace.store.save(workspace.controller.getSnapshot())
  })

  ipcMain.handle(IPC_CHANNELS.workspaceSetBrowserUrl, async (_event, url: string) => {
    const workspace = input.requireWorkspace()
    const normalized = normalizeUrl(url)
    workspace.controller.setBrowserUrl(normalized)
    await workspace.browserView.webContents.loadURL(normalized)
    await workspace.store.save(workspace.controller.getSnapshot())
  })

  ipcMain.handle(IPC_CHANNELS.workspaceGoBack, async () => {
    const workspace = input.requireWorkspace()

    if (!workspace.browserView.webContents.canGoBack()) {
      return
    }

    workspace.browserView.webContents.goBack()
  })

  ipcMain.handle(IPC_CHANNELS.workspaceGoForward, async () => {
    const workspace = input.requireWorkspace()

    if (!workspace.browserView.webContents.canGoForward()) {
      return
    }

    workspace.browserView.webContents.goForward()
  })

  ipcMain.handle(IPC_CHANNELS.workspaceAdjustSplitter, async (_event, delta: number) => {
    const workspace = input.requireWorkspace()
    const current = workspace.controller.getSnapshot()
    workspace.controller.setNotesWidth(current.notesWidth + delta)
    await workspace.store.save(workspace.controller.getSnapshot())
  })

  ipcMain.handle(IPC_CHANNELS.opencodeGetState, async () => {
    const openCode = input.getOpenCode()
    if (!openCode) {
      return createDefaultOpenCodeState(input.openCodeRepoRoot)
    }

    try {
      return await openCode.service.initialize()
    } catch {
      return openCode.service.getState()
    }
  })

  ipcMain.handle(IPC_CHANNELS.opencodeSendPrompt, async (_event, prompt: string) => {
    const openCode = input.requireOpenCode()
    await openCode.service.sendPrompt(prompt)
  })

  ipcMain.handle(IPC_CHANNELS.terminalGetState, async () => {
    const terminal = input.getTerminal()
    if (!terminal) {
      return createDefaultTerminalState(input.terminalRepoRoot, resolveTerminalShell())
    }

    return terminal.service.getState()
  })

  ipcMain.handle(IPC_CHANNELS.terminalConnect, async (_event, cols: number, rows: number) => {
    const terminal = input.requireTerminal()
    await terminal.service.connect(cols, rows)
  })

  ipcMain.handle(IPC_CHANNELS.terminalWrite, async (_event, data: string) => {
    const terminal = input.requireTerminal()
    await terminal.service.write(data)
  })

  ipcMain.handle(IPC_CHANNELS.terminalResize, async (_event, cols: number, rows: number) => {
    const terminal = input.requireTerminal()
    await terminal.service.resize(cols, rows)
  })

  ipcMain.handle(IPC_CHANNELS.terminalRestart, async (_event, cols: number, rows: number) => {
    const terminal = input.requireTerminal()
    await terminal.service.restart(cols, rows)
  })
}
