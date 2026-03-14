import { spawn as nodeSpawn, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { setTimeout as delay } from 'node:timers/promises'

import type { BrowserContextSnapshot } from '../../../browser-model'
import { createDefaultOpenCodeState, type OpenCodeMessage, type OpenCodeState } from '../../../opencode-model'

type FetchLike = typeof fetch

type SpawnLike = typeof nodeSpawn

type ServerHandle = {
  baseUrl: string
  child: ChildProcess
}

type BrowserMcpConfig = {
  url: string
  headers: Record<string, string>
}

export function buildOpenCodeConfig(input?: { browserMcp?: BrowserMcpConfig }): string {
  const mcp = input?.browserMcp
    ? {
        browser: {
          type: 'remote',
          url: input.browserMcp.url,
          headers: input.browserMcp.headers,
          oauth: false,
          enabled: true,
        },
      }
    : undefined

  return JSON.stringify({
    $schema: 'https://opencode.ai/config.json',
    default_agent: 'plan',
    share: 'disabled',
    ...(mcp ? { mcp } : {}),
    permission: {
      '*': 'deny',
      read: 'allow',
      glob: 'allow',
      grep: 'allow',
      list: 'allow',
      lsp: 'allow',
      edit: 'deny',
      bash: 'deny',
      task: 'deny',
      skill: 'deny',
      webfetch: 'deny',
      websearch: 'deny',
      todoread: 'deny',
      todowrite: 'deny',
      external_directory: 'deny',
      'browser_*': 'allow',
    },
  })
}

async function findAvailablePort(): Promise<number> {
  const { createServer } = await import('node:net')

  return await new Promise<number>((resolve, reject) => {
    const server = createServer()

    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        server.close(() => reject(new Error('Could not reserve an OpenCode port')))
        return
      }

      const { port } = address
      server.close((error) => {
        if (error) {
          reject(error)
          return
        }

        resolve(port)
      })
    })
  })
}

function createMessage(role: OpenCodeMessage['role'], text: string): OpenCodeMessage {
  return {
    id: randomUUID(),
    role,
    text,
  }
}

function extractAssistantText(payload: { parts?: Array<{ type?: string; text?: string }> }): string {
  const text = (payload.parts ?? [])
    .filter((part) => part.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text?.trim() ?? '')
    .filter(Boolean)
    .join('\n\n')

  return text || 'OpenCode responded without any plain-text output.'
}

async function requestJson<T>(fetchImpl: FetchLike, url: string, init?: RequestInit): Promise<T> {
  const response = await fetchImpl(url, {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    throw new Error(`OpenCode request failed with status ${response.status}`)
  }

  return (await response.json()) as T
}

export class OpenCodeService {
  private readonly listeners = new Set<(state: OpenCodeState) => void>()
  private state: OpenCodeState
  private server: ServerHandle | null = null
  private initialization: Promise<OpenCodeState> | null = null
  private disposed = false

  constructor(
    private readonly input: {
      repoRoot: string
      mockMode?: boolean
      fetchImpl?: FetchLike
      spawn?: SpawnLike
      getPort?: () => Promise<number>
      environment?: NodeJS.ProcessEnv
      browserMcp?: BrowserMcpConfig
      browserContextProvider?: () => Promise<BrowserContextSnapshot | null>
    },
  ) {
    this.state = createDefaultOpenCodeState(input.repoRoot)
  }

  getState(): OpenCodeState {
    return this.state
  }

  subscribe(listener: (state: OpenCodeState) => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  async initialize(): Promise<OpenCodeState> {
    if (this.state.sessionId && this.state.status !== 'error') {
      return this.state
    }

    if (this.initialization) {
      return this.initialization
    }

    this.updateState({ status: 'connecting', error: null })

    this.initialization = (this.input.mockMode ?? process.env.ELECTRON_OPENCODE_MOCK === '1')
      ? this.initializeMock()
      : this.initializeServerBacked()

    try {
      return await this.initialization
    } finally {
      this.initialization = null
    }
  }

  async sendPrompt(prompt: string): Promise<void> {
    const trimmed = prompt.trim()
    if (!trimmed) {
      return
    }

    await this.initialize()

    const nextMessages = [...this.state.messages, createMessage('user', trimmed)]
    this.updateState({ messages: nextMessages, status: 'responding', error: null })

    try {
      const assistantReply = (this.input.mockMode ?? process.env.ELECTRON_OPENCODE_MOCK === '1')
        ? await this.buildMockReply(trimmed)
        : await this.requestAssistantReply(trimmed)

      this.updateState({
        messages: [...this.state.messages, createMessage('assistant', assistantReply)],
        status: 'ready',
        error: null,
      })
    } catch (error) {
      this.updateState({
        status: 'error',
        error: error instanceof Error ? error.message : 'OpenCode could not answer the prompt.',
      })
      throw error
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true
    if (this.server) {
      this.server.child.kill()
      this.server = null
    }
  }

  private async initializeMock(): Promise<OpenCodeState> {
    this.updateState({
      sessionId: 'mock-session',
      status: 'ready',
      error: null,
    })

    return this.state
  }

  private async initializeServerBacked(): Promise<OpenCodeState> {
    const fetchImpl = this.input.fetchImpl ?? fetch
    const server = await this.ensureServer()
    await this.waitForHealth(server.baseUrl, fetchImpl)

    const session = await requestJson<{ id: string }>(fetchImpl, `${server.baseUrl}/session`, {
      method: 'POST',
      body: JSON.stringify({ title: 'Electron OpenCode Chat' }),
    })

    this.updateState({
      sessionId: session.id,
      status: 'ready',
      error: null,
    })

    await requestJson(fetchImpl, `${server.baseUrl}/session/${session.id}/message`, {
      method: 'POST',
      body: JSON.stringify({
        noReply: true,
        parts: [
          {
            type: 'text',
            text:
              'When the user asks what is visible in the browser, what page is on screen, or asks you to explain the current browser page, call the browser_browser_context_current tool before answering.',
          },
        ],
      }),
    })

    return this.state
  }

  private async requestAssistantReply(prompt: string): Promise<string> {
    const sessionId = this.state.sessionId
    if (!sessionId || !this.server) {
      throw new Error('OpenCode session is not ready')
    }

    const fetchImpl = this.input.fetchImpl ?? fetch
    const result = await requestJson<{ parts?: Array<{ type?: string; text?: string }> }>(
      fetchImpl,
      `${this.server.baseUrl}/session/${sessionId}/message`,
      {
        method: 'POST',
        body: JSON.stringify({
          agent: 'plan',
          parts: [{ type: 'text', text: prompt }],
        }),
      },
    )

    return extractAssistantText(result)
  }

  private async buildMockReply(prompt: string): Promise<string> {
    if (/what do you see in the browser|what is in the browser|explain the browser/i.test(prompt)) {
      return await this.buildMockBrowserReply()
    }

    return `Mock OpenCode reply for ${this.input.repoRoot}: ${prompt}`
  }

  private async buildMockBrowserReply(): Promise<string> {
    const browserContext = await this.input.browserContextProvider?.().catch(() => null)
    if (!browserContext) {
      return 'I cannot inspect the browser on the right right now because its live page context is unavailable.'
    }

    return (
      `I can see the browser is currently at ${browserContext.url}. ` +
      `I also captured a fresh screenshot of the browser pane (${browserContext.screenshot.width}x${browserContext.screenshot.height}), ` +
      'so this explanation is based on that screenshot.'
    )
  }

  private async ensureServer(): Promise<ServerHandle> {
    if (this.server) {
      return this.server
    }

    const port = await (this.input.getPort ?? findAvailablePort)()
    const child = (this.input.spawn ?? nodeSpawn)(
      'opencode',
      ['serve', '--hostname', '127.0.0.1', '--port', String(port)],
      {
        cwd: this.input.repoRoot,
        stdio: 'ignore',
        env: {
          ...process.env,
          ...(this.input.environment ?? {}),
          OPENCODE_CONFIG_CONTENT: buildOpenCodeConfig({
            browserMcp: this.input.browserMcp,
          }),
        },
      },
    )

    child.once('exit', () => {
      if (this.disposed) {
        return
      }

      this.server = null
      this.updateState({
        status: 'error',
        error: 'The local OpenCode server stopped unexpectedly.',
      })
    })

    this.server = {
      baseUrl: `http://127.0.0.1:${port}`,
      child,
    }

    return this.server
  }

  private async waitForHealth(baseUrl: string, fetchImpl: FetchLike): Promise<void> {
    let lastError: unknown

    for (let attempt = 0; attempt < 20; attempt += 1) {
      try {
        await requestJson(fetchImpl, `${baseUrl}/global/health`)
        return
      } catch (error) {
        lastError = error
        await delay(250)
      }
    }

    throw lastError instanceof Error ? lastError : new Error('OpenCode server did not become healthy in time')
  }

  private updateState(partial: Partial<OpenCodeState>): void {
    this.state = {
      ...this.state,
      ...partial,
    }

    for (const listener of this.listeners) {
      listener(this.state)
    }
  }
}
