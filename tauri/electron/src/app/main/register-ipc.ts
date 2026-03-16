import type { BrowserHost } from './browser-host'
import { ipcMain } from 'electron'

import { DEFAULT_BROWSER_SNAPSHOT } from '../../browser-model'
import { createDefaultOpenCodeState } from '../../opencode-model'
import { IPC_CHANNELS } from '../../ipc'
import { DEFAULT_WORKSPACE_SNAPSHOT } from '../../workspace-model'
import type { OpenCodeBundle } from './create-opencode-window'
import type { WorkspaceBundle } from './create-workspace-window'

export function registerIpc(input: {
  createWorkspace: () => Promise<void>
  createOpenCode: () => Promise<void>
  getWorkspace: () => WorkspaceBundle | null
  getOpenCode: () => OpenCodeBundle | null
  requireWorkspace: () => WorkspaceBundle
  requireOpenCode: () => OpenCodeBundle
  getBrowserHostForSender: (webContentsId: number) => BrowserHost | null
  openCodeRepoRoot: string
}) {
  const requireBrowserHost = (senderId: number) => {
    const host = input.getBrowserHostForSender(senderId)
    if (!host) {
      throw new Error('Browser surface is not available for this renderer')
    }

    return host
  }

  ipcMain.handle(IPC_CHANNELS.launcherOpenWorkspace, async () => {
    await input.createWorkspace()
  })

  ipcMain.handle(IPC_CHANNELS.launcherOpenOpenCode, async () => {
    await input.createOpenCode()
  })

  ipcMain.handle(IPC_CHANNELS.browserGetState, async (event) => {
    const host = input.getBrowserHostForSender(event.sender.id)
    return host ? host.getSnapshot() : DEFAULT_BROWSER_SNAPSHOT
  })

  ipcMain.handle(IPC_CHANNELS.browserSetUrl, async (event, url: string) => {
    await requireBrowserHost(event.sender.id).setBrowserUrl(url)
  })

  ipcMain.handle(IPC_CHANNELS.browserGoBack, async (event) => {
    requireBrowserHost(event.sender.id).goBack()
  })

  ipcMain.handle(IPC_CHANNELS.browserGoForward, async (event) => {
    requireBrowserHost(event.sender.id).goForward()
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

  ipcMain.handle(IPC_CHANNELS.opencodeAdjustSplit, async (_event, delta: number) => {
    input.requireOpenCode().controller.adjustOpenCodeWidth(delta)
  })
}
