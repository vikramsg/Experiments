import path from 'node:path'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        launcher: path.resolve(import.meta.dirname, 'launcher.html'),
        notes: path.resolve(import.meta.dirname, 'notes.html'),
        splitter: path.resolve(import.meta.dirname, 'splitter.html'),
      },
    },
  },
})
