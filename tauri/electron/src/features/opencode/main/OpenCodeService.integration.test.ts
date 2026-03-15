// @vitest-environment node

import { BrowserMcpServer } from '../../browser/main/BrowserMcpServer'
import { OpenCodeService } from './OpenCodeService'

function hasOpencodeBinary() {
  return typeof process.env.PATH === 'string' && process.env.PATH.length > 0
}

describe.sequential('OpenCodeService real MCP integration', () => {
  const repoRoot = '/Users/vikramsingh/Projects/Personal/Experiments/tauri'

  const maybeIt = hasOpencodeBinary() ? it : it.skip

  maybeIt('connects the browser MCP server and registers the browser tool in OpenCode', async () => {
    const browserMcpServer = new BrowserMcpServer({
      getBrowserContext: vi.fn().mockResolvedValue({
        url: 'https://example.com/integration',
        screenshot: {
          mimeType: 'image/png',
          data: Buffer.from('integration-png').toString('base64'),
          width: 1280,
          height: 720,
        },
      }),
    })
    const browserMcp = await browserMcpServer.start()
    const service = new OpenCodeService({
      repoRoot,
      browserMcp,
    })

    try {
      await service.initialize()

      const baseUrl = service.getServerBaseUrl()
      expect(baseUrl).not.toBeNull()

      const mcpStatusResponse = await fetch(`${baseUrl}/mcp`)
      expect(mcpStatusResponse.ok).toBe(true)

      const mcpStatus = (await mcpStatusResponse.json()) as Record<string, { status: string }>
      expect(mcpStatus.browser?.status).toBe('connected')

    } finally {
      await service.dispose()
      await browserMcpServer.stop()
    }
  }, 120000)

  maybeIt('lets real OpenCode tool execution invoke the browser MCP tool', async () => {
    const browserMcpServer = new BrowserMcpServer({
      getBrowserContext: vi.fn().mockResolvedValue({
        url: 'https://example.com/tool-call',
        screenshot: {
          mimeType: 'image/png',
          data: Buffer.from('tool-call-png').toString('base64'),
          width: 1024,
          height: 768,
        },
      }),
    })
    const browserMcp = await browserMcpServer.start()

    try {
      const service = new OpenCodeService({
        repoRoot,
        browserMcp,
        model: 'azure-openai/gpt-5-4',
      })

      await service.initialize()
      const baseUrl = service.getServerBaseUrl()
      expect(baseUrl).not.toBeNull()
      const sessionId = service.getState().sessionId
      expect(sessionId).not.toBeNull()

      const response = await fetch(`${baseUrl}/session/${sessionId}/message`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          agent: 'plan',
          parts: [
            {
              type: 'text',
              text:
                'Call the browser_browser_context_current tool exactly once and then answer with the current browser URL only.',
            },
          ],
        }),
        signal: AbortSignal.timeout(60000),
      })

      expect(response.ok).toBe(true)

      expect(browserMcpServer.getToolCallCount()).toBeGreaterThan(0)

      await service.dispose()
    } finally {
      await browserMcpServer.stop()
    }
  }, 120000)
})
