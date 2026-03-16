import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { BUNDLED_TERMINAL_FONT_FAMILY, DEFAULT_TERMINAL_APPEARANCE } from '../../../terminal-model'

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

  it('uses the bundled render font on first load and logs when Ghostty config differs', async () => {
    await writeFile(ghosttyConfigPath, 'font-family = JetBrains Mono\n', 'utf8')
    const logger = { warn: vi.fn() }

    const store = new TerminalAppearanceStore(userDataPath, ghosttyConfigPath, logger)

    await expect(store.load()).resolves.toEqual({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: BUNDLED_TERMINAL_FONT_FAMILY,
    })
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({
        ghosttyFontFamily: 'JetBrains Mono',
        renderFontFamily: BUNDLED_TERMINAL_FONT_FAMILY,
      }),
      expect.stringContaining('Ghostty terminal font differs'),
    )
    await expect(readFile(join(userDataPath, 'terminal-appearance.json'), 'utf8')).resolves.toContain(BUNDLED_TERMINAL_FONT_FAMILY)
  })

  it('upgrades old saved appearance files to the bundled render font while preserving size and chrome settings', async () => {
    await writeFile(ghosttyConfigPath, 'font-family = JetBrains Mono\n', 'utf8')
    await writeFile(
      join(userDataPath, 'terminal-appearance.json'),
      JSON.stringify({
        fontFamily: 'Iosevka Term',
        fontSize: 15,
        minimalChrome: false,
      }),
      'utf8',
    )
    const logger = { warn: vi.fn() }

    const store = new TerminalAppearanceStore(userDataPath, ghosttyConfigPath, logger)

    await expect(store.load()).resolves.toEqual({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: BUNDLED_TERMINAL_FONT_FAMILY,
      fontSize: 15,
      minimalChrome: false,
    })
  })

  it('does not log when Ghostty config is missing and still uses the bundled render font', async () => {
    const logger = { warn: vi.fn() }
    const store = new TerminalAppearanceStore(userDataPath, ghosttyConfigPath, logger)

    await expect(store.load()).resolves.toEqual(DEFAULT_TERMINAL_APPEARANCE)
    expect(logger.warn).not.toHaveBeenCalled()
  })
})
