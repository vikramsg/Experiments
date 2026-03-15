import { DEFAULT_TERMINAL_APPEARANCE, resolveTerminalAppearance, toGhosttyWebFontFamily } from '../../../terminal-model'

describe('terminal appearance helpers', () => {
  it('prefers saved Electron appearance over imported Ghostty defaults', () => {
    expect(
      resolveTerminalAppearance({
        saved: {
          ...DEFAULT_TERMINAL_APPEARANCE,
          fontFamily: 'Iosevka Term',
        },
        importedFontFamily: 'JetBrains Mono',
      }),
    ).toEqual({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: 'Iosevka Term',
    })
  })

  it('uses imported Ghostty font-family when no saved Electron preference exists', () => {
    expect(resolveTerminalAppearance({ importedFontFamily: 'JetBrains Mono' })).toEqual({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: 'JetBrains Mono',
    })
  })

  it('builds a Ghostty/Web font stack with symbol fallbacks for prompt glyphs', () => {
    expect(toGhosttyWebFontFamily('JetBrains Mono')).toBe(
      'JetBrains Mono, Symbols Nerd Font Mono, Symbols Nerd Font, Apple Color Emoji, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace',
    )
  })
})
