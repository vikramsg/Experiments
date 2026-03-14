export type PermissionSessionLike = {
  setPermissionRequestHandler: (
    handler: (_webContents: unknown, _permission: string, callback: (allowed: boolean) => void) => void,
  ) => void
}

export type BrowserWebContentsLike = {
  setWindowOpenHandler: (handler: () => { action: 'deny' | 'allow' }) => void
  on: (event: 'will-navigate', listener: (event: { preventDefault: () => void }, url: string) => void) => void
}

export function normalizeUrl(url: string) {
  if (/^[a-z][a-z\d+.-]*:/i.test(url) && !/^https?:\/\//i.test(url)) {
    throw new Error('Only http and https URLs are allowed')
  }

  const candidate = /^https?:\/\//i.test(url) ? url : `https://${url}`
  const parsed = new URL(candidate)

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('Only http and https URLs are allowed')
  }

  return parsed.toString()
}

export function applyBrowserSecurityPolicy(input: {
  session: PermissionSessionLike
  webContents: BrowserWebContentsLike
}) {
  input.session.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false)
  })

  input.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  input.webContents.on('will-navigate', (event, url) => {
    try {
      normalizeUrl(url)
    } catch {
      event.preventDefault()
    }
  })
}
