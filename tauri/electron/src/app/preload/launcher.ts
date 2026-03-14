import { contextBridge, ipcRenderer } from 'electron'

import { IPC_CHANNELS } from '../../shared/ipc/channels'

contextBridge.exposeInMainWorld('launcher', {
  openWorkspace: () => ipcRenderer.invoke(IPC_CHANNELS.launcherOpenWorkspace) as Promise<void>,
})
