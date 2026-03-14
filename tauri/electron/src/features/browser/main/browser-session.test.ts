import { applyBrowserSecurityPolicy, normalizeUrl } from './browser-session'

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
})
