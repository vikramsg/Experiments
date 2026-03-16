import { EventEmitter } from 'node:events'

import { OpenCodeService, buildOpenCodeConfig } from './OpenCodeService'

function createJsonResponse(data: unknown): Response {
  return new Response(JSON.stringify(data), {
    headers: { 'content-type': 'application/json' },
  })
}

describe('OpenCodeService', () => {
  it('builds a read-only OpenCode config with the browser MCP tool enabled', () => {
    const config = JSON.parse(
      buildOpenCodeConfig({
        browserMcp: {
          url: 'http://127.0.0.1:4318/mcp',
          headers: { authorization: 'Bearer test-token' },
        },
      }),
    ) as {
      default_agent: string
      share: string
      permission: Record<string, string>
      mcp: Record<string, { type: string; url: string; headers?: Record<string, string> }>
    }

    expect(config.default_agent).toBe('plan')
    expect(config.share).toBe('disabled')
    expect(config.permission.read).toBe('allow')
    expect(config.permission.glob).toBe('allow')
    expect(config.permission.grep).toBe('allow')
    expect(config.permission.list).toBe('allow')
    expect(config.permission.edit).toBe('deny')
    expect(config.permission.bash).toBe('deny')
    expect(config.permission['browser_*']).toBe('allow')
    expect(config.mcp.browser).toEqual({
      type: 'remote',
      url: 'http://127.0.0.1:4318/mcp',
      headers: { authorization: 'Bearer test-token' },
      oauth: false,
      enabled: true,
    })
  })

  it('starts the local server and creates a repo-scoped session', async () => {
    const child = new EventEmitter() as EventEmitter & { kill: ReturnType<typeof vi.fn> }
    child.kill = vi.fn()

    const spawn = vi.fn().mockReturnValue(child)
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(createJsonResponse({ healthy: true, version: '1.0.0' }))
      .mockResolvedValueOnce(createJsonResponse({ id: 'session-1' }))
      .mockResolvedValueOnce(createJsonResponse({ info: { id: 'message-1' }, parts: [] }))
      .mockResolvedValueOnce(createJsonResponse({ browser: { status: 'connected' } }))

    const service = new OpenCodeService({
      repoRoot: '/repo/tauri',
      mockMode: false,
      spawn,
      fetchImpl: fetchMock,
      getPort: vi.fn().mockResolvedValue(4173),
      browserMcp: {
        url: 'http://127.0.0.1:4318/mcp',
        headers: { authorization: 'Bearer test-token' },
      },
    })

    await expect(service.initialize()).resolves.toMatchObject({
      status: 'ready',
      repoRoot: '/repo/tauri',
      sessionId: 'session-1',
      browserToolStatus: 'ready',
    })

    expect(spawn).toHaveBeenCalledWith(
      'opencode',
      ['serve', '--hostname', '127.0.0.1', '--port', '4173'],
      expect.objectContaining({ cwd: '/repo/tauri' }),
    )
  })

  it('sends prompts and appends user plus assistant messages', async () => {
    const child = new EventEmitter() as EventEmitter & { kill: ReturnType<typeof vi.fn> }
    child.kill = vi.fn()

    const spawn = vi.fn().mockReturnValue(child)
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(createJsonResponse({ healthy: true, version: '1.0.0' }))
      .mockResolvedValueOnce(createJsonResponse({ id: 'session-1' }))
      .mockResolvedValueOnce(createJsonResponse({ info: { id: 'message-1' }, parts: [] }))
      .mockResolvedValueOnce(createJsonResponse({ browser: { status: 'connected' } }))
      .mockResolvedValueOnce(
        createJsonResponse({
          parts: [
            { type: 'text', text: 'This repo contains an Electron app under tauri/electron.' },
          ],
        }),
      )

    const service = new OpenCodeService({
      repoRoot: '/repo/tauri',
      mockMode: false,
      spawn,
      fetchImpl: fetchMock,
      getPort: vi.fn().mockResolvedValue(4174),
      browserMcp: {
        url: 'http://127.0.0.1:4318/mcp',
        headers: { authorization: 'Bearer test-token' },
      },
    })

    await service.initialize()
    await service.sendPrompt('What is in this repo?')

    expect(service.getState().messages.at(-2)).toMatchObject({ role: 'user', text: 'What is in this repo?' })
    expect(service.getState().messages.at(-1)).toMatchObject({
      role: 'assistant',
      text: 'This repo contains an Electron app under tauri/electron.',
    })
  })

  it('verifies browser MCP availability before marking browser-aware mode ready', async () => {
    const child = new EventEmitter() as EventEmitter & { kill: ReturnType<typeof vi.fn> }
    child.kill = vi.fn()

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(createJsonResponse({ healthy: true, version: '1.0.0' }))
      .mockResolvedValueOnce(createJsonResponse({ id: 'session-1' }))
      .mockResolvedValueOnce(createJsonResponse({ info: { id: 'message-1' }, parts: [] }))
      .mockResolvedValueOnce(createJsonResponse({ browser: { status: 'failed', error: 'handshake failed' } }))

    const service = new OpenCodeService({
      repoRoot: '/repo/tauri',
      mockMode: false,
      spawn: vi.fn().mockReturnValue(child),
      fetchImpl: fetchMock,
      getPort: vi.fn().mockResolvedValue(4175),
      browserMcp: {
        url: 'http://127.0.0.1:4318/mcp',
        headers: { authorization: 'Bearer test-token' },
      },
    })

    await service.initialize()

    expect(service.getState()).toMatchObject({
      browserToolStatus: 'unavailable',
      browserToolMessage: expect.stringMatching(/handshake failed/i),
    })
  })

  it('retries browser-aware prompts with a stronger tool instruction and errors if the tool is never called', async () => {
    const child = new EventEmitter() as EventEmitter & { kill: ReturnType<typeof vi.fn> }
    child.kill = vi.fn()

    const toolCallCount = 0
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(createJsonResponse({ healthy: true, version: '1.0.0' }))
      .mockResolvedValueOnce(createJsonResponse({ id: 'session-1' }))
      .mockResolvedValueOnce(createJsonResponse({ info: { id: 'message-1' }, parts: [] }))
      .mockResolvedValueOnce(createJsonResponse({ browser: { status: 'connected' } }))
      .mockResolvedValueOnce(createJsonResponse({ parts: [{ type: 'text', text: 'I do not have the tool.' }] }))
      .mockResolvedValueOnce(createJsonResponse({ parts: [{ type: 'text', text: 'Still no tool usage.' }] }))

    const service = new OpenCodeService({
      repoRoot: '/repo/tauri',
      mockMode: false,
      spawn: vi.fn().mockReturnValue(child),
      fetchImpl: fetchMock,
      getPort: vi.fn().mockResolvedValue(4176),
      browserMcp: {
        url: 'http://127.0.0.1:4318/mcp',
        headers: { authorization: 'Bearer test-token' },
      },
      getBrowserToolCallCount: () => toolCallCount,
    })

    await service.initialize()

    await expect(service.sendPrompt('What do you see in the browser?')).rejects.toThrow(/did not invoke the browser context tool/i)

    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      'http://127.0.0.1:4176/session/session-1/message',
      expect.objectContaining({
        body: expect.stringContaining('Before answering, call the browser_browser_context_current tool'),
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      'http://127.0.0.1:4176/session/session-1/message',
      expect.objectContaining({
        body: expect.stringContaining('You must call the browser_browser_context_current tool exactly once'),
      }),
    )
  })
})
