import { ipcMain } from 'electron'

import { normalizeUrl } from '../../features/browser/main/browser-session'
import { IPC_CHANNELS } from '../../ipc'
import { DEFAULT_WORKSPACE_SNAPSHOT } from '../../workspace-model'
import type { WorkspaceBundle } from './create-workspace-window'

export function registerIpc(input: {
  createWorkspace: () => Promise<void>
  getWorkspace: () => WorkspaceBundle | null
  requireWorkspace: () => WorkspaceBundle
}) {
  ipcMain.handle(IPC_CHANNELS.launcherOpenWorkspace, async () => {
    await input.createWorkspace()
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
}
