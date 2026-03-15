import { contextBridge, ipcRenderer } from 'electron'

import { IPC_CHANNELS } from '../../ipc'
import type { LauncherApi } from '../../workspace-contract'

const launcherApi: LauncherApi = {
  openWorkspace: () => ipcRenderer.invoke(IPC_CHANNELS.launcherOpenWorkspace) as Promise<void>,
  openOpenCode: () => ipcRenderer.invoke(IPC_CHANNELS.launcherOpenOpenCode) as Promise<void>,
  openTerminal: () => ipcRenderer.invoke(IPC_CHANNELS.launcherOpenTerminal) as Promise<void>,
}

contextBridge.exposeInMainWorld('launcher', launcherApi)
