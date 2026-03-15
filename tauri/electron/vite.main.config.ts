import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    sourcemap: true,
    rollupOptions: {
      external: ['@lydell/node-pty', /^@lydell\/node-pty-/],
    },
  },
})
