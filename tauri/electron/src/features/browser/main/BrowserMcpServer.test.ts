import { BrowserMcpServer } from './BrowserMcpServer'

describe('BrowserMcpServer', () => {
  it('returns text metadata plus a screenshot attachment for browser context', async () => {
    const server = new BrowserMcpServer({
      getBrowserContext: vi.fn().mockResolvedValue({
        url: 'https://example.com/docs',
        screenshot: {
          mimeType: 'image/png',
          data: Buffer.from('png-bytes').toString('base64'),
          width: 1280,
          height: 720,
        },
      }),
    })

    await expect(server.handleBrowserContextTool()).resolves.toMatchObject({
      content: [
        expect.objectContaining({ type: 'text' }),
        expect.objectContaining({ type: 'image', mimeType: 'image/png' }),
      ],
    })
  })

  it('returns a graceful text-only result when the browser is unavailable', async () => {
    const server = new BrowserMcpServer({
      getBrowserContext: vi.fn().mockResolvedValue(null),
    })

    await expect(server.handleBrowserContextTool()).resolves.toMatchObject({
      content: [
        expect.objectContaining({
          type: 'text',
          text: expect.stringMatching(/browser surface for this window is not available/i),
        }),
      ],
    })
  })
})
