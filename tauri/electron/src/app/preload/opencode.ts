import { contextBridge, ipcRenderer } from 'electron'

import { IPC_CHANNELS } from '../../ipc'
import type { OpenCodeApi } from '../../opencode-contract'
import type { OpenCodeState } from '../../opencode-model'

const openCodeApi: OpenCodeApi = {
  loadState: () => ipcRenderer.invoke(IPC_CHANNELS.opencodeGetState) as Promise<OpenCodeState>,
  sendPrompt: (prompt: string) => ipcRenderer.invoke(IPC_CHANNELS.opencodeSendPrompt, prompt) as Promise<void>,
  adjustSplit: (delta: number) => ipcRenderer.invoke(IPC_CHANNELS.opencodeAdjustSplit, delta) as Promise<void>,
  onStateChange: (listener: (state: OpenCodeState) => void) => {
    const wrapped = (_event: Electron.IpcRendererEvent, state: OpenCodeState) => {
      listener(state)
    }

    ipcRenderer.on(IPC_CHANNELS.opencodeState, wrapped)
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.opencodeState, wrapped)
    }
  },
}

contextBridge.exposeInMainWorld('opencode', openCodeApi)
