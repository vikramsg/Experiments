import {
  BUNDLED_TERMINAL_FONT_FAMILY,
  DEFAULT_TERMINAL_APPEARANCE,
  resolveTerminalAppearance,
  toGhosttyWebFontFamily,
} from '../../../terminal-model'

describe('terminal appearance helpers', () => {
  it('upgrades saved appearance to the bundled render font while preserving size and chrome settings', () => {
    expect(
      resolveTerminalAppearance({
        saved: {
          fontFamily: 'Iosevka Term',
          fontSize: 15,
          minimalChrome: false,
        },
      }),
    ).toEqual({
      ...DEFAULT_TERMINAL_APPEARANCE,
      fontFamily: BUNDLED_TERMINAL_FONT_FAMILY,
      fontSize: 15,
      minimalChrome: false,
    })
  })

  it('keeps the bundled render font when there is no saved Electron preference', () => {
    expect(resolveTerminalAppearance({})).toEqual(DEFAULT_TERMINAL_APPEARANCE)
  })

  it('builds a Ghostty/Web font family from the single bundled render font only', () => {
    expect(toGhosttyWebFontFamily(DEFAULT_TERMINAL_APPEARANCE)).toBe(BUNDLED_TERMINAL_FONT_FAMILY)
  })
})
