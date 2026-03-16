import { OpenCodeBrowserController } from './OpenCodeBrowserController'

function createView() {
  return {
    setBounds: vi.fn(),
    webContents: {
      close: vi.fn(),
    },
  }
}

describe('OpenCodeBrowserController', () => {
  it('lays out the OpenCode pane, splitter, and browser panes', () => {
    const openCodeView = createView()
    const splitterView = createView()
    const browserChromeView = createView()
    const browserView = createView()

    new OpenCodeBrowserController(
      {
        getContentBounds: () => ({ width: 1200, height: 900 }),
        on: vi.fn(),
      },
      {
        openCodeView,
        splitterView,
        browserChromeView,
        browserView,
      },
    )

    expect(openCodeView.setBounds).toHaveBeenCalled()
    expect(splitterView.setBounds).toHaveBeenCalled()
    expect(browserChromeView.setBounds).toHaveBeenCalled()
    expect(browserView.setBounds).toHaveBeenCalled()
  })
})
