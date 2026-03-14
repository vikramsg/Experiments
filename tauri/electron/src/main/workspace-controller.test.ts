import { WorkspaceController, type BrowserViewLike, type Rectangle, type WorkspaceWindowLike } from './workspace-controller'

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
  it('applies initial browser bounds from the saved splitter width', () => {
    const browserBounds: Rectangle[] = []
    const browserView: BrowserViewLike = {
      setBounds: (bounds) => browserBounds.push(bounds),
      webContents: {
        close: vi.fn(),
      },
    }

    const windowMock = createWindowMock()

    new WorkspaceController(windowMock.window, browserView, {
      notes: 'hello',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    expect(browserBounds.at(-1)).toEqual({ x: 432, y: 72, width: 768, height: 728 })
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
    const browserBounds: Rectangle[] = []
    const browserView: BrowserViewLike = {
      setBounds: (bounds) => browserBounds.push(bounds),
      webContents: {
        close: vi.fn(),
      },
    }

    const windowMock = createWindowMock()
    const controller = new WorkspaceController(windowMock.window, browserView, {
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    controller.setNotesWidth(520)

    expect(controller.getSnapshot().notesWidth).toBe(520)
    expect(browserBounds.at(-1)).toEqual({ x: 532, y: 72, width: 668, height: 728 })
  })

  it('recomputes bounds on window resize', () => {
    const browserBounds: Rectangle[] = []
    const browserView: BrowserViewLike = {
      setBounds: (bounds) => browserBounds.push(bounds),
      webContents: {
        close: vi.fn(),
      },
    }

    const windowMock = createWindowMock()
    const controller = new WorkspaceController(windowMock.window, browserView, {
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    windowMock.window.getContentBounds = () => ({ width: 1400, height: 900 })
    windowMock.emit('resize')

    expect(controller.getSnapshot().notesWidth).toBe(420)
    expect(browserBounds.at(-1)).toEqual({ x: 432, y: 72, width: 968, height: 828 })
  })

  it('closes the browser webContents when the workspace closes', () => {
    const browserView: BrowserViewLike = {
      setBounds: vi.fn(),
      webContents: {
        close: vi.fn(),
      },
    }

    const windowMock = createWindowMock()

    new WorkspaceController(windowMock.window, browserView, {
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    windowMock.emit('closed')

    expect(browserView.webContents.close).toHaveBeenCalledTimes(1)
  })
})
