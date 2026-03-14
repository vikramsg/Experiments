import { ipcMain } from 'electron'

import { normalizeUrl } from '../../features/browser/main/browser-session'
import { IPC_CHANNELS } from '../../shared/ipc/channels'
import type { WorkspaceBundle } from './create-workspace-window'

export function registerIpc(input: {
  createWorkspace: () => Promise<void>
  requireWorkspace: () => WorkspaceBundle
}) {
  ipcMain.handle(IPC_CHANNELS.launcherOpenWorkspace, async () => {
    await input.createWorkspace()
  })

  ipcMain.handle(IPC_CHANNELS.workspaceGetState, async () => {
    const workspace = input.requireWorkspace()
    return workspace.controller.getSnapshot()
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

  ipcMain.handle(IPC_CHANNELS.workspaceAdjustSplitter, async (_event, delta: number) => {
    const workspace = input.requireWorkspace()
    const current = workspace.controller.getSnapshot()
    workspace.controller.setNotesWidth(current.notesWidth + delta)
    await workspace.store.save(workspace.controller.getSnapshot())
  })
}
