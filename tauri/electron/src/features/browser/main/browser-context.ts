import { normalizeUrl } from './browser-session'

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

export type BrowserContextWebContentsLike = {
  getURL: () => string
  capturePage: () => Promise<{
    toPNG: () => Buffer
    getSize: () => { width: number; height: number }
  }>
}

export async function getBrowserContextSnapshot(
  webContents: BrowserContextWebContentsLike,
): Promise<BrowserContextSnapshot> {
  const url = normalizeUrl(webContents.getURL())
  const screenshot = await webContents.capturePage()
  const size = screenshot.getSize()

  return {
    url,
    screenshot: {
      mimeType: 'image/png',
      data: screenshot.toPNG().toString('base64'),
      width: size.width,
      height: size.height,
    },
  }
}
