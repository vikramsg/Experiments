import { contextBridge, ipcRenderer } from 'electron'

import { IPC_CHANNELS } from '../../ipc'
import type { TerminalApi } from '../../terminal-contract'
import type { TerminalState } from '../../terminal-model'

const terminalApi: TerminalApi = {
  loadState: () => ipcRenderer.invoke(IPC_CHANNELS.terminalGetState) as Promise<TerminalState>,
  connect: (cols: number, rows: number) => ipcRenderer.invoke(IPC_CHANNELS.terminalConnect, cols, rows) as Promise<void>,
  write: (data: string) => ipcRenderer.invoke(IPC_CHANNELS.terminalWrite, data) as Promise<void>,
  resize: (cols: number, rows: number) => ipcRenderer.invoke(IPC_CHANNELS.terminalResize, cols, rows) as Promise<void>,
  restart: (cols: number, rows: number) => ipcRenderer.invoke(IPC_CHANNELS.terminalRestart, cols, rows) as Promise<void>,
  onData: (listener: (data: string) => void) => {
    const wrapped = (_event: Electron.IpcRendererEvent, data: string) => {
      listener(data)
    }

    ipcRenderer.on(IPC_CHANNELS.terminalData, wrapped)
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.terminalData, wrapped)
    }
  },
  onStateChange: (listener: (state: TerminalState) => void) => {
    const wrapped = (_event: Electron.IpcRendererEvent, state: TerminalState) => {
      listener(state)
    }

    ipcRenderer.on(IPC_CHANNELS.terminalState, wrapped)
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.terminalState, wrapped)
    }
  },
}

contextBridge.exposeInMainWorld('terminal', terminalApi)
