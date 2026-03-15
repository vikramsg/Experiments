import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { randomUUID } from 'node:crypto'
import { setTimeout as delay } from 'node:timers/promises'

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js'
import { z } from 'zod'

import type { BrowserContextSnapshot } from './browser-context'

type CallToolResult = {
  content: Array<
    | { type: 'text'; text: string }
    | { type: 'image'; data: string; mimeType: string }
  >
}

export type BrowserMcpConnection = {
  url: string
  headers: Record<string, string>
}

async function findAvailablePort(): Promise<number> {
  const { createServer } = await import('node:net')

  return await new Promise<number>((resolve, reject) => {
    const server = createServer()

    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        server.close(() => reject(new Error('Could not reserve a browser MCP port')))
        return
      }

      server.close((error) => {
        if (error) {
          reject(error)
          return
        }

        resolve(address.port)
      })
    })
  })
}

async function readRequestBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = []

  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }

  if (chunks.length === 0) {
    return undefined
  }

  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

export class BrowserMcpServer {
  private readonly token: string
  private readonly mcpServer: McpServer
  private readonly transport: StreamableHTTPServerTransport
  private httpServer: Server | null = null
  private connection: BrowserMcpConnection | null = null
  private toolCallCount = 0

  constructor(
    private readonly input: {
      getBrowserContext: () => Promise<BrowserContextSnapshot | null>
      getPort?: () => Promise<number>
      token?: string
    },
  ) {
    this.token = input.token ?? randomUUID()
    this.mcpServer = new McpServer({
      name: 'electron-browser-context',
      version: '0.1.0',
    })
    this.transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
    })

    this.mcpServer.registerTool(
      'browser_context_current',
      {
        description:
          'Inspect the current browser page by returning the current URL and a fresh screenshot of the browser pane.',
        inputSchema: z.object({}),
      },
      async () => this.handleBrowserContextTool(),
    )
  }

  async start(): Promise<BrowserMcpConnection> {
    if (this.connection) {
      return this.connection
    }

    const port = await (this.input.getPort ?? findAvailablePort)()
    await this.mcpServer.connect(this.transport)

    this.httpServer = createServer(async (request, response) => {
      await this.handleHttpRequest(request, response)
    })

    await new Promise<void>((resolve, reject) => {
      this.httpServer?.once('error', reject)
      this.httpServer?.listen(port, '127.0.0.1', () => resolve())
    })

    this.connection = {
      url: `http://127.0.0.1:${port}/mcp`,
      headers: {
        authorization: `Bearer ${this.token}`,
      },
    }

    return this.connection
  }

  async stop(): Promise<void> {
    if (!this.httpServer && !this.connection) {
      return
    }

    const closeHttpServer = this.httpServer
      ? new Promise<void>((resolve, reject) => {
          this.httpServer?.closeAllConnections?.()
          this.httpServer?.close((error) => {
            if (error) {
              reject(error)
              return
            }

            resolve()
          })
        })
      : Promise.resolve()

    await closeHttpServer
    this.httpServer = null
    this.connection = null
    await Promise.race([this.transport.close(), delay(5000)])
    await Promise.race([this.mcpServer.close(), delay(5000)])
  }

  async handleBrowserContextTool(): Promise<CallToolResult> {
    try {
      this.toolCallCount += 1
      const browserContext = await this.input.getBrowserContext()

      if (!browserContext) {
        return {
          content: [
            {
              type: 'text',
              text: 'The browser surface for this window is not available right now, so there is no live browser context to inspect.',
            },
          ],
        }
      }

      return {
        content: [
          {
            type: 'text',
            text:
              `Current browser URL: ${browserContext.url}\n` +
              `Fresh screenshot captured from the browser pane (${browserContext.screenshot.width}x${browserContext.screenshot.height}).`,
          },
          {
            type: 'image',
            data: browserContext.screenshot.data,
            mimeType: browserContext.screenshot.mimeType,
          },
        ],
      }
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text:
              error instanceof Error
                ? `The browser screenshot could not be captured: ${error.message}`
                : 'The browser screenshot could not be captured.',
          },
        ],
      }
    }
  }

  getToolCallCount(): number {
    return this.toolCallCount
  }

  resetToolCallCount(): void {
    this.toolCallCount = 0
  }

  private async handleHttpRequest(request: IncomingMessage, response: ServerResponse) {
    if (!request.url?.startsWith('/mcp')) {
      response.statusCode = 404
      response.end('Not found')
      return
    }

    if (request.headers.authorization !== `Bearer ${this.token}`) {
      response.statusCode = 401
      response.end('Unauthorized')
      return
    }

    const parsedBody = request.method === 'POST' ? await readRequestBody(request) : undefined
    await this.transport.handleRequest(request, response, parsedBody)
  }
}
