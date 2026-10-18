import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

function autoStartNotifierPlugin() {
  return {
    name: 'auto-start-notifier',
    configureServer() {
      try {
        const notifierPath = path.resolve(__dirname, '../notifier.py')
        console.log('[Vite] Auto-starting background Alert Monitor (notifier.py)...')
        const p = spawn('python', [notifierPath], {
          detached: true,
          stdio: 'ignore',
          windowsHide: true
        })
        p.unref()
      } catch (err) {
        console.error('[Vite] Error auto-starting notifier:', err)
      }
    }
  }
}

export default defineConfig({
  plugins: [react(), autoStartNotifierPlugin()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8766',
        changeOrigin: true,
      },
    },
  },
})
