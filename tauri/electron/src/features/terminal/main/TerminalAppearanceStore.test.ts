import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { DEFAULT_TERMINAL_APPEARANCE } from '../../../terminal-model'

import { TerminalAppearanceStore } from './TerminalAppearanceStore'

describe('TerminalAppearanceStore', () => {
  let userDataPath: string
  let ghosttyConfigPath: string

  beforeEach(async () => {
    userDataPath = await mkdtemp(join(tmpdir(), 'electron-terminal-appearance-'))
    ghosttyConfigPath = join(userDataPath, 'ghostty-config')
  })

  afterEach(async () => {
    await rm(userDataPath, { recursive: true, force: true })
  })

  it('imports Ghostty font-family on first load and persists the seeded appearance', async () => {
    await writeFile(ghosttyConfigPath, 'font-family = JetBrains Mono\n', 'utf8')

    const store = new TerminalAppearanceStore(userDataPath, ghosttyConfigPath)

    await expect(store.load()).resolves.toEqual({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: 'JetBrains Mono',
    })
    await expect(readFile(join(userDataPath, 'terminal-appearance.json'), 'utf8')).resolves.toContain('JetBrains Mono')
  })

  it('keeps the saved Electron appearance instead of re-importing Ghostty defaults later', async () => {
    await writeFile(ghosttyConfigPath, 'font-family = JetBrains Mono\n', 'utf8')

    const store = new TerminalAppearanceStore(userDataPath, ghosttyConfigPath)
    await store.save({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: 'Iosevka Term',
      fontSize: 15,
    })

    await expect(store.load()).resolves.toEqual({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: 'Iosevka Term',
      fontSize: 15,
    })
  })

  it('falls back to durable defaults when Ghostty config is missing or malformed', async () => {
    const store = new TerminalAppearanceStore(userDataPath, ghosttyConfigPath)

    await expect(store.load()).resolves.toEqual(DEFAULT_TERMINAL_APPEARANCE)
  })
})
