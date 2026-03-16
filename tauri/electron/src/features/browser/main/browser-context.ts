import { normalizeUrl } from './browser-session'
import type { BrowserContextSnapshot } from '../../../browser-model'

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
