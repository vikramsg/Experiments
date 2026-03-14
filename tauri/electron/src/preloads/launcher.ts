import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('launcher', {
  openWorkspace: () => ipcRenderer.invoke('launcher:open-workspace') as Promise<void>,
})
