import { contextBridge, ipcRenderer } from 'electron'

import type { WorkspaceSnapshot } from '../main/note-store'

contextBridge.exposeInMainWorld('workspace', {
  loadState: () => ipcRenderer.invoke('workspace:get-state') as Promise<WorkspaceSnapshot>,
  saveNotes: (notes: string) => ipcRenderer.invoke('workspace:save-notes', notes) as Promise<void>,
  setBrowserUrl: (url: string) => ipcRenderer.invoke('workspace:set-browser-url', url) as Promise<void>,
  adjustSplitter: (delta: number) => ipcRenderer.invoke('workspace:adjust-splitter', delta) as Promise<void>,
  onStateChange: (listener: (snapshot: WorkspaceSnapshot) => void) => {
    const wrapped = (_event: Electron.IpcRendererEvent, snapshot: WorkspaceSnapshot) => {
      listener(snapshot)
    }

    ipcRenderer.on('workspace:state', wrapped)
    return () => {
      ipcRenderer.removeListener('workspace:state', wrapped)
    }
  },
})
