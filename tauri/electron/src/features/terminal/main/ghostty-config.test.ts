import { parseGhosttyConfigFontFamily } from './ghostty-config'

describe('parseGhosttyConfigFontFamily', () => {
  it('extracts the active font-family from Ghostty config text', () => {
    expect(
      parseGhosttyConfigFontFamily(`
# comment
shell-integration-features = ssh-terminfo
font-family = JetBrains Mono
`),
    ).toBe('JetBrains Mono')
  })

  it('ignores commented font-family lines and returns null when missing', () => {
    expect(
      parseGhosttyConfigFontFamily(`
# font-family = Wrong Font
theme = tokyonight
`),
    ).toBeNull()
  })
})
