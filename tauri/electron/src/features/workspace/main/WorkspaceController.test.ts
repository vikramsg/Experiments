import { WorkspaceController, type Rectangle, type ViewLike, type WorkspaceWindowLike } from './WorkspaceController'

function createWindowMock() {
  const listeners: Partial<Record<'resize' | 'closed', Array<() => void>>> = {}
  const sends: Array<{ channel: string; payload: unknown }> = []

  const window: WorkspaceWindowLike = {
    getContentBounds: () => ({ width: 1200, height: 800 }),
    on: (event, listener) => {
      listeners[event] ??= []
      listeners[event]!.push(listener)
    },
    webContents: {
      send: (channel, payload) => {
        sends.push({ channel, payload })
      },
    },
  }

  return {
    window,
    sends,
    emit(event: 'resize' | 'closed') {
      for (const listener of listeners[event] ?? []) {
        listener()
      }
    },
  }
}

describe('WorkspaceController', () => {
  function createViewMock() {
    const bounds: Rectangle[] = []
    const view: ViewLike = {
      setBounds: (nextBounds) => bounds.push(nextBounds),
      webContents: {
        close: vi.fn(),
      },
    }

    return { bounds, view }
  }

  it('applies initial bounds to all four sibling views from the saved splitter width', () => {
    const notesView = createViewMock()
    const splitterView = createViewMock()
    const browserChromeView = createViewMock()
    const browserView = createViewMock()

    const windowMock = createWindowMock()

    new WorkspaceController(
      windowMock.window,
      {
        notesView: notesView.view,
        splitterView: splitterView.view,
        browserChromeView: browserChromeView.view,
        browserView: browserView.view,
      },
      {
        notes: 'hello',
        notesWidth: 420,
        browserUrl: 'https://example.com',
      },
    )

    expect(notesView.bounds.at(-1)).toEqual({ x: 0, y: 0, width: 420, height: 800 })
    expect(splitterView.bounds.at(-1)).toEqual({ x: 420, y: 0, width: 12, height: 800 })
    expect(browserChromeView.bounds.at(-1)).toEqual({ x: 432, y: 0, width: 768, height: 64 })
    expect(browserView.bounds.at(-1)).toEqual({ x: 432, y: 64, width: 768, height: 736 })
    expect(windowMock.sends.at(-1)).toEqual({
      channel: 'workspace:state',
      payload: {
        notes: 'hello',
        notesWidth: 420,
        browserUrl: 'https://example.com',
      },
    })
  })

  it('updates bounds when the splitter moves', () => {
    const notesView = createViewMock()
    const splitterView = createViewMock()
    const browserChromeView = createViewMock()
    const browserView = createViewMock()

    const windowMock = createWindowMock()
    const controller = new WorkspaceController(
      windowMock.window,
      {
        notesView: notesView.view,
        splitterView: splitterView.view,
        browserChromeView: browserChromeView.view,
        browserView: browserView.view,
      },
      {
        notes: '',
        notesWidth: 420,
        browserUrl: 'https://example.com',
      },
    )

    controller.setNotesWidth(520)

    expect(controller.getSnapshot().notesWidth).toBe(520)
    expect(notesView.bounds.at(-1)).toEqual({ x: 0, y: 0, width: 520, height: 800 })
    expect(splitterView.bounds.at(-1)).toEqual({ x: 520, y: 0, width: 12, height: 800 })
    expect(browserChromeView.bounds.at(-1)).toEqual({ x: 532, y: 0, width: 668, height: 64 })
    expect(browserView.bounds.at(-1)).toEqual({ x: 532, y: 64, width: 668, height: 736 })
  })

  it('recomputes bounds on window resize', () => {
    const notesView = createViewMock()
    const splitterView = createViewMock()
    const browserChromeView = createViewMock()
    const browserView = createViewMock()

    const windowMock = createWindowMock()
    const controller = new WorkspaceController(
      windowMock.window,
      {
        notesView: notesView.view,
        splitterView: splitterView.view,
        browserChromeView: browserChromeView.view,
        browserView: browserView.view,
      },
      {
        notes: '',
        notesWidth: 420,
        browserUrl: 'https://example.com',
      },
    )

    windowMock.window.getContentBounds = () => ({ width: 1400, height: 900 })
    windowMock.emit('resize')

    expect(controller.getSnapshot().notesWidth).toBe(420)
    expect(notesView.bounds.at(-1)).toEqual({ x: 0, y: 0, width: 420, height: 900 })
    expect(splitterView.bounds.at(-1)).toEqual({ x: 420, y: 0, width: 12, height: 900 })
    expect(browserChromeView.bounds.at(-1)).toEqual({ x: 432, y: 0, width: 968, height: 64 })
    expect(browserView.bounds.at(-1)).toEqual({ x: 432, y: 64, width: 968, height: 836 })
  })

  it('closes every child webContents when the workspace closes', () => {
    const notesView = createViewMock()
    const splitterView = createViewMock()
    const browserChromeView = createViewMock()
    const browserView = createViewMock()

    const windowMock = createWindowMock()

    new WorkspaceController(
      windowMock.window,
      {
        notesView: notesView.view,
        splitterView: splitterView.view,
        browserChromeView: browserChromeView.view,
        browserView: browserView.view,
      },
      {
        notes: '',
        notesWidth: 420,
        browserUrl: 'https://example.com',
      },
    )

    windowMock.emit('closed')

    expect(notesView.view.webContents.close).toHaveBeenCalledTimes(1)
    expect(splitterView.view.webContents.close).toHaveBeenCalledTimes(1)
    expect(browserChromeView.view.webContents.close).toHaveBeenCalledTimes(1)
    expect(browserView.view.webContents.close).toHaveBeenCalledTimes(1)
  })
})
