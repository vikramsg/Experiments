import { applyBrowserSecurityPolicy, normalizeUrl, readBrowserNavigationState, subscribeToBrowserNavigation } from './browser-session'

describe('browser-session', () => {
  it('normalizes hostnames to https URLs', () => {
    expect(normalizeUrl('example.com')).toBe('https://example.com/')
  })

  it('rejects non-http protocols', () => {
    expect(() => normalizeUrl('file:///tmp/test')).toThrow(/only http and https/i)
  })

  it('denies all permission requests and new windows', () => {
    const callbacks: boolean[] = []
    const permissionSession = {
      setPermissionRequestHandler: vi.fn((handler: (_wc: unknown, _permission: string, cb: (allowed: boolean) => void) => void) => {
        handler({}, 'notifications', (allowed) => callbacks.push(allowed))
      }),
    }

    const webContents = {
      setWindowOpenHandler: vi.fn(),
      on: vi.fn(),
    }

    applyBrowserSecurityPolicy({
      session: permissionSession,
      webContents,
    })

    expect(permissionSession.setPermissionRequestHandler).toHaveBeenCalledTimes(1)
    expect(callbacks).toEqual([false])
    expect(webContents.setWindowOpenHandler).toHaveBeenCalledWith(expect.any(Function))
    expect(webContents.on).toHaveBeenCalledWith('will-navigate', expect.any(Function))
  })

  it('reads the current browser navigation state from webContents', () => {
    const webContents = {
      getURL: vi.fn().mockReturnValue('https://example.org/docs'),
      canGoBack: vi.fn().mockReturnValue(true),
      canGoForward: vi.fn().mockReturnValue(false),
    }

    expect(readBrowserNavigationState(webContents)).toEqual({
      browserUrl: 'https://example.org/docs',
      canGoBack: true,
      canGoForward: false,
    })
  })

  it('subscribes to browser navigation events and publishes fresh browser state', () => {
    const listeners: Record<string, Array<(...args: unknown[]) => void>> = {
      'did-navigate': [],
      'did-navigate-in-page': [],
    }

    const webContents = {
      getURL: vi.fn().mockReturnValue('https://example.com/'),
      canGoBack: vi.fn().mockReturnValue(false),
      canGoForward: vi.fn().mockReturnValue(false),
      on: vi.fn((event: 'did-navigate' | 'did-navigate-in-page', listener: (...args: unknown[]) => void) => {
        listeners[event].push(listener)
      }),
    }

    const onChange = vi.fn()
    subscribeToBrowserNavigation({ webContents, onChange })

    webContents.getURL.mockReturnValue('https://www.iana.org/help/example-domains')
    webContents.canGoBack.mockReturnValue(true)

    listeners['did-navigate'][0]({}, 'https://www.iana.org/help/example-domains')

    expect(onChange).toHaveBeenCalledWith({
      browserUrl: 'https://www.iana.org/help/example-domains',
      canGoBack: true,
      canGoForward: false,
    })
    expect(webContents.on).toHaveBeenCalledWith('did-navigate', expect.any(Function))
    expect(webContents.on).toHaveBeenCalledWith('did-navigate-in-page', expect.any(Function))
  })
})
