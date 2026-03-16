import { spawn as nodeSpawn, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { setTimeout as delay } from 'node:timers/promises'

import type { BrowserContextSnapshot } from '../../../browser-model'
import { createDefaultOpenCodeState, type OpenCodeMessage, type OpenCodeState } from '../../../opencode-model'

const EMPTY_ASSISTANT_TEXT = 'OpenCode responded without any plain-text output.'

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

type BrowserToolVerification = {
  connected: boolean
  registered: boolean
  message: string | null
}

export function buildOpenCodeConfig(input?: { browserMcp?: BrowserMcpConfig; model?: string }): string {
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
    ...(input?.model ? { model: input.model } : {}),
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

  return text || EMPTY_ASSISTANT_TEXT
}

function extractAssistantTextFromMessages(
  messages: Array<{
    info?: { role?: string }
    parts?: Array<{ type?: string; text?: string; ignored?: boolean }>
  }>,
): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.info?.role !== 'assistant') {
      continue
    }

    const text = (message.parts ?? [])
      .filter((part) => part.type === 'text' && typeof part.text === 'string' && !part.ignored)
      .map((part) => part.text?.trim() ?? '')
      .filter(Boolean)
      .join('\n\n')

    if (text) {
      return text
    }
  }

  return EMPTY_ASSISTANT_TEXT
}

async function waitForAssistantTextFromMessages(
  fetchImpl: FetchLike,
  baseUrl: string,
  sessionId: string,
  timeoutMs: number,
): Promise<string> {
  const startedAt = Date.now()

  while (Date.now() - startedAt < timeoutMs) {
    const messages = await requestJson<
      Array<{
        info?: { role?: string }
        parts?: Array<{ type?: string; text?: string; ignored?: boolean }>
      }>
    >(fetchImpl, `${baseUrl}/session/${sessionId}/message`, {
      timeoutMs: 15000,
    })

    const assistantText = extractAssistantTextFromMessages(messages)
    if (assistantText !== EMPTY_ASSISTANT_TEXT) {
      return assistantText
    }

    await delay(500)
  }

  return EMPTY_ASSISTANT_TEXT
}

async function requestJson<T>(
  fetchImpl: FetchLike,
  url: string,
  init?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const request = fetchImpl(url, {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  const response = await Promise.race<Response>([
    request,
    delay(init?.timeoutMs ?? 10000).then<Response>(() => {
      throw new Error(`OpenCode request timed out after ${init?.timeoutMs ?? 10000}ms`)
    }),
  ])

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
      getBrowserToolCallCount?: () => number
      model?: string
    },
  ) {
    this.state = createDefaultOpenCodeState(input.repoRoot)
  }

  getState(): OpenCodeState {
    return this.state
  }

  getServerBaseUrl(): string | null {
    return this.server?.baseUrl ?? null
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

    const browserAware = this.isBrowserAwarePrompt(trimmed)
    if (browserAware && this.state.browserToolStatus !== 'ready') {
      this.updateState({
        messages: [
          ...this.state.messages,
          createMessage('user', trimmed),
          createMessage(
            'assistant',
            this.state.browserToolMessage ??
              'Browser inspection is not available in this OpenCode session right now.',
          ),
        ],
        status: 'ready',
      })
      return
    }

    const nextMessages = [...this.state.messages, createMessage('user', trimmed)]
    this.updateState({ messages: nextMessages, status: 'responding', error: null })

    try {
      const beforeBrowserToolCalls = browserAware && this.input.getBrowserToolCallCount
        ? this.input.getBrowserToolCallCount()
        : null

      const assistantReply = (this.input.mockMode ?? process.env.ELECTRON_OPENCODE_MOCK === '1')
        ? await this.buildMockReply(trimmed)
        : await this.requestBrowserAwareReply(trimmed, browserAware, beforeBrowserToolCalls)

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
      const child = this.server.child
      const exited = new Promise<void>((resolve) => {
        child.once('exit', () => resolve())
      })

      child.kill()
      const exitedGracefully = await Promise.race([exited.then(() => true), delay(5000).then(() => false)])
      if (!exitedGracefully) {
        child.kill('SIGKILL')
        await Promise.race([exited, delay(2000).then(() => undefined)])
      }
      this.server = null
    }
  }

  private async initializeMock(): Promise<OpenCodeState> {
    const browserToolAvailable = Boolean(this.input.browserContextProvider)
    this.updateState({
      sessionId: 'mock-session',
      status: 'ready',
      error: null,
      browserToolStatus: browserToolAvailable ? 'ready' : 'unavailable',
      browserToolMessage: browserToolAvailable ? 'Browser inspection is available in mock mode.' : 'Browser inspection is unavailable in mock mode.',
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
      timeoutMs: 15000,
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
      timeoutMs: 30000,
    })

    const browserToolVerification = await this.verifyBrowserTooling(server.baseUrl, fetchImpl)

    this.updateState({
      sessionId: session.id,
      status: 'ready',
      error: browserToolVerification.connected && browserToolVerification.registered ? null : browserToolVerification.message,
      browserToolStatus:
        browserToolVerification.connected && browserToolVerification.registered ? 'ready' : 'unavailable',
      browserToolMessage: browserToolVerification.message,
      messages:
        browserToolVerification.connected && browserToolVerification.registered
          ? this.state.messages
          : [
              ...this.state.messages,
              createMessage(
                'system',
                browserToolVerification.message ??
                  'Browser inspection is unavailable, so browser-aware questions may not work in this session.',
              ),
            ],
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
        timeoutMs: 60000,
      },
    )

    const directText = extractAssistantText(result)
    if (directText !== EMPTY_ASSISTANT_TEXT) {
      return directText
    }

    return await waitForAssistantTextFromMessages(fetchImpl, this.server.baseUrl, sessionId, 30000)
  }

  private async requestBrowserAwareReply(
    prompt: string,
    browserAware: boolean,
    beforeBrowserToolCalls: number | null,
  ): Promise<string> {
    if (!browserAware) {
      return await this.requestAssistantReply(prompt)
    }

    const firstAttempt = await this.requestAssistantReply(this.buildBrowserAwarePrompt(prompt, false))
    if (this.browserToolWasUsed(beforeBrowserToolCalls) && firstAttempt !== EMPTY_ASSISTANT_TEXT) {
      return firstAttempt
    }

    if (this.browserToolWasUsed(beforeBrowserToolCalls)) {
      const summaryAttempt = await this.requestAssistantReply(
        `Using the browser context tool result you already retrieved, answer the user's request in plain text. User request: ${prompt}`,
      )
      if (summaryAttempt !== EMPTY_ASSISTANT_TEXT) {
        return summaryAttempt
      }
    }

    const retryAttempt = await this.requestAssistantReply(this.buildBrowserAwarePrompt(prompt, true))
    if (this.browserToolWasUsed(beforeBrowserToolCalls) && retryAttempt !== EMPTY_ASSISTANT_TEXT) {
      return retryAttempt
    }

    if (this.browserToolWasUsed(beforeBrowserToolCalls)) {
      const summaryAttempt = await this.requestAssistantReply(
        `Using the browser context tool result you already retrieved, answer the user's request in plain text. User request: ${prompt}`,
      )
      if (summaryAttempt !== EMPTY_ASSISTANT_TEXT) {
        return summaryAttempt
      }
    }

    throw new Error('OpenCode did not invoke the browser context tool for this request.')
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

  private async verifyBrowserTooling(baseUrl: string, fetchImpl: FetchLike): Promise<BrowserToolVerification> {
    if (!this.input.browserMcp) {
      return {
        connected: false,
        registered: false,
        message: 'Browser MCP is not configured for this OpenCode session.',
      }
    }

    const mcpStatus = await requestJson<Record<string, { status?: string; error?: string }>>(fetchImpl, `${baseUrl}/mcp`, {
      timeoutMs: 5000,
    })
    const browserStatus = mcpStatus.browser
    if (browserStatus?.status !== 'connected') {
      return {
        connected: false,
        registered: false,
        message: browserStatus?.error
          ? `Browser MCP failed to connect: ${browserStatus.error}`
          : 'Browser MCP is not connected in this OpenCode session.',
      }
    }

    return {
      connected: true,
      registered: true,
      message: 'Browser inspection is ready for this session.',
    }
  }

  private isBrowserAwarePrompt(prompt: string): boolean {
    return /what do you see in the browser|what is in the browser|explain the browser|what is currently showing up in the browser/i.test(prompt)
  }

  private buildBrowserAwarePrompt(prompt: string, strict: boolean): string {
    const prefix = strict
      ? 'You must call the browser_browser_context_current tool exactly once before answering. Use the tool result, including the screenshot, to answer the user. '
      : 'Before answering, call the browser_browser_context_current tool and use its URL and screenshot result to answer the user. '

    return `${prefix}User request: ${prompt}`
  }

  private browserToolWasUsed(beforeCallCount: number | null): boolean {
    if (beforeCallCount === null || !this.input.getBrowserToolCallCount) {
      return false
    }

    return this.input.getBrowserToolCallCount() > beforeCallCount
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
            model: this.input.model,
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
        await requestJson(fetchImpl, `${baseUrl}/global/health`, { timeoutMs: 5000 })
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
