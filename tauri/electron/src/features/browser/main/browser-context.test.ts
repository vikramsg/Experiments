import { getBrowserContextSnapshot } from './browser-context'

describe('browser-context', () => {
  it('reads the browser URL and serializes a screenshot attachment', async () => {
    const webContents = {
      getURL: vi.fn().mockReturnValue('https://example.com/docs'),
      capturePage: vi.fn().mockResolvedValue({
        toPNG: () => Buffer.from('png-bytes'),
        getSize: () => ({ width: 800, height: 600 }),
      }),
    }

    await expect(getBrowserContextSnapshot(webContents)).resolves.toMatchObject({
      url: 'https://example.com/docs',
      screenshot: {
        mimeType: 'image/png',
        data: Buffer.from('png-bytes').toString('base64'),
        width: 800,
        height: 600,
      },
    })
  })

  it('fails predictably when a screenshot cannot be captured', async () => {
    const webContents = {
      getURL: vi.fn().mockReturnValue('https://example.com/docs'),
      capturePage: vi.fn().mockRejectedValue(new Error('capture failed')),
    }

    await expect(getBrowserContextSnapshot(webContents)).rejects.toThrow(/capture failed/i)
  })
})
