import path from 'node:path'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        launcher: path.resolve(import.meta.dirname, 'src/app/renderer/entries/launcher.html'),
        notes: path.resolve(import.meta.dirname, 'src/app/renderer/entries/notes.html'),
        splitter: path.resolve(import.meta.dirname, 'src/app/renderer/entries/splitter.html'),
        'browser-chrome': path.resolve(import.meta.dirname, 'src/app/renderer/entries/browser-chrome.html'),
        opencode: path.resolve(import.meta.dirname, 'src/app/renderer/entries/opencode.html'),
        terminal: path.resolve(import.meta.dirname, 'src/app/renderer/entries/terminal.html'),
      },
    },
  },
})
