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

    const service = new OpenCodeService({
      repoRoot: '/repo/tauri',
      mockMode: false,
      spawn,
      fetchImpl: fetchMock,
      getPort: vi.fn().mockResolvedValue(4173),
    })

    await expect(service.initialize()).resolves.toMatchObject({
      status: 'ready',
      repoRoot: '/repo/tauri',
      sessionId: 'session-1',
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
    })

    await service.initialize()
    await service.sendPrompt('What is in this repo?')

    expect(service.getState().messages.at(-2)).toMatchObject({ role: 'user', text: 'What is in this repo?' })
    expect(service.getState().messages.at(-1)).toMatchObject({
      role: 'assistant',
      text: 'This repo contains an Electron app under tauri/electron.',
    })
  })
})
