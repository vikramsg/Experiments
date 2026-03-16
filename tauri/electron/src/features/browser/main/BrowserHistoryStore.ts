import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { normalizeUrl } from './browser-session'

const MAX_BROWSER_HISTORY = 10

export class BrowserHistoryStore {
  private readonly filePath: string
  private history: string[] = []
  private loaded = false
  private readonly listeners = new Set<(history: string[]) => void>()

  constructor(userDataPath: string) {
    this.filePath = join(userDataPath, 'browser-history.json')
  }

  async load(): Promise<void> {
    if (this.loaded) {
      return
    }

    try {
      const raw = await readFile(this.filePath, 'utf8')
      const parsed = JSON.parse(raw) as { history?: string[] }
      this.history = Array.isArray(parsed.history) ? parsed.history.slice(0, MAX_BROWSER_HISTORY) : []
    } catch {
      this.history = []
    }

    this.loaded = true
  }

  getHistory(): string[] {
    return [...this.history]
  }

  subscribe(listener: (history: string[]) => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  async remember(url: string): Promise<void> {
    await this.load()

    // This history is only for browser input autocomplete. MCP/browser context
    // stays limited to the current live page rather than recent navigation.
    const normalized = normalizeUrl(url)
    this.history = [normalized, ...this.history.filter((entry) => entry !== normalized)].slice(0, MAX_BROWSER_HISTORY)
    await mkdir(dirname(this.filePath), { recursive: true })
    await writeFile(this.filePath, JSON.stringify({ history: this.history }, null, 2), 'utf8')

    for (const listener of this.listeners) {
      listener(this.getHistory())
    }
  }
}
