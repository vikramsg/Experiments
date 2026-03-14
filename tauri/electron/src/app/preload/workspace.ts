import { contextBridge, ipcRenderer } from 'electron'

import { IPC_CHANNELS } from '../../shared/ipc/channels'
import type { WorkspaceSnapshot } from '../../shared/types/workspace'

contextBridge.exposeInMainWorld('workspace', {
  loadState: () => ipcRenderer.invoke(IPC_CHANNELS.workspaceGetState) as Promise<WorkspaceSnapshot>,
  saveNotes: (notes: string) => ipcRenderer.invoke(IPC_CHANNELS.workspaceSaveNotes, notes) as Promise<void>,
  setBrowserUrl: (url: string) => ipcRenderer.invoke(IPC_CHANNELS.workspaceSetBrowserUrl, url) as Promise<void>,
  adjustSplitter: (delta: number) => ipcRenderer.invoke(IPC_CHANNELS.workspaceAdjustSplitter, delta) as Promise<void>,
  onStateChange: (listener: (snapshot: WorkspaceSnapshot) => void) => {
    const wrapped = (_event: Electron.IpcRendererEvent, snapshot: WorkspaceSnapshot) => {
      listener(snapshot)
    }

    ipcRenderer.on(IPC_CHANNELS.workspaceState, wrapped)
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.workspaceState, wrapped)
    }
  },
})
