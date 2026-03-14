import path from 'node:path'

import { BrowserWindow } from 'electron'

function getLauncherEntryUrl() {
  if (LAUNCHER_WINDOW_VITE_DEV_SERVER_URL) {
    return new URL('src/app/renderer/entries/launcher.html', LAUNCHER_WINDOW_VITE_DEV_SERVER_URL).toString()
  }

  return path.join(__dirname, `../renderer/${LAUNCHER_WINDOW_VITE_NAME}/src/app/renderer/entries/launcher.html`)
}

export function createLauncherWindow() {
  const launcherWindow = new BrowserWindow({
    width: 1100,
    height: 760,
    title: 'Electron Workspace',
    webPreferences: {
      preload: path.join(__dirname, 'launcher.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const target = getLauncherEntryUrl()
  if (target.startsWith('http')) {
    void launcherWindow.loadURL(target)
  } else {
    void launcherWindow.loadFile(target)
  }

  return launcherWindow
}
