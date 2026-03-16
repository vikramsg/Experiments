/**
 * Canonical browser navigation model shared across local browser chrome
 * renderers and the main-process browser host.
 *
 * This file stays at the root because both Browser + Notes and OpenCode +
 * Browser rely on the same browser snapshot shape without importing each other.
 * It is a domain boundary, not business logic.
 */
export type BrowserSnapshot = {
  browserUrl: string
  canGoBack: boolean
  canGoForward: boolean
  recentUrls: string[]
}

export type BrowserScreenshot = {
  mimeType: 'image/png'
  data: string
  width: number
  height: number
}

export type BrowserContextSnapshot = {
  url: string
  screenshot: BrowserScreenshot
}

export const DEFAULT_BROWSER_SNAPSHOT: BrowserSnapshot = {
  browserUrl: 'https://example.com',
  canGoBack: false,
  canGoForward: false,
  recentUrls: [],
}
