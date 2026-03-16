import { contextBridge, ipcRenderer } from 'electron'

import { IPC_CHANNELS } from '../../ipc'
import type { BrowserApi } from '../../browser-contract'
import type { BrowserSnapshot } from '../../browser-model'

const browserApi: BrowserApi = {
  loadState: () => ipcRenderer.invoke(IPC_CHANNELS.browserGetState) as Promise<BrowserSnapshot>,
  setBrowserUrl: (url: string) => ipcRenderer.invoke(IPC_CHANNELS.browserSetUrl, url) as Promise<void>,
  goBack: () => ipcRenderer.invoke(IPC_CHANNELS.browserGoBack) as Promise<void>,
  goForward: () => ipcRenderer.invoke(IPC_CHANNELS.browserGoForward) as Promise<void>,
  onStateChange: (listener: (snapshot: BrowserSnapshot) => void) => {
    const wrapped = (_event: Electron.IpcRendererEvent, snapshot: BrowserSnapshot) => {
      listener(snapshot)
    }

    ipcRenderer.on(IPC_CHANNELS.browserState, wrapped)
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.browserState, wrapped)
    }
  },
}

contextBridge.exposeInMainWorld('browser', browserApi)
