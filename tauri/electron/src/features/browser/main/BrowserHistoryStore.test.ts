import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BrowserHistoryStore } from './BrowserHistoryStore'

describe('BrowserHistoryStore', () => {
  it('keeps the latest 10 visited urls in most-recent-first order', async () => {
    const userDataDir = await mkdtemp(join(tmpdir(), 'browser-history-store-'))

    try {
      const store = new BrowserHistoryStore(userDataDir)
      for (let index = 0; index < 12; index += 1) {
        await store.remember(`https://example.com/${index}`)
      }

      expect(store.getHistory()).toEqual([
        'https://example.com/11',
        'https://example.com/10',
        'https://example.com/9',
        'https://example.com/8',
        'https://example.com/7',
        'https://example.com/6',
        'https://example.com/5',
        'https://example.com/4',
        'https://example.com/3',
        'https://example.com/2',
      ])
    } finally {
      await rm(userDataDir, { recursive: true, force: true })
    }
  })

  it('dedupes urls and reloads persisted history', async () => {
    const userDataDir = await mkdtemp(join(tmpdir(), 'browser-history-store-'))

    try {
      const firstStore = new BrowserHistoryStore(userDataDir)
      await firstStore.remember('https://example.com/one')
      await firstStore.remember('https://example.com/two')
      await firstStore.remember('https://example.com/one')

      const secondStore = new BrowserHistoryStore(userDataDir)
      await secondStore.load()

      expect(secondStore.getHistory()).toEqual([
        'https://example.com/one',
        'https://example.com/two',
      ])
    } finally {
      await rm(userDataDir, { recursive: true, force: true })
    }
  })
})
