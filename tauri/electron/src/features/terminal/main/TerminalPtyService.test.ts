import { TerminalPtyService } from './TerminalPtyService'

type MockPtyProcess = {
  write: ReturnType<typeof vi.fn>
  resize: ReturnType<typeof vi.fn>
  kill: ReturnType<typeof vi.fn>
  onData: (listener: (data: string) => void) => { dispose: () => void }
  onExit: (listener: (event: { exitCode: number; signal?: number }) => void) => { dispose: () => void }
  emitData: (data: string) => void
  emitExit: (event: { exitCode: number; signal?: number }) => void
}

function createMockPtyProcess(): MockPtyProcess {
  let dataListener: ((data: string) => void) | undefined
  let exitListener: ((event: { exitCode: number; signal?: number }) => void) | undefined

  return {
    write: vi.fn(),
    resize: vi.fn(),
    kill: vi.fn(),
    onData: (listener) => {
      dataListener = listener
      return { dispose: () => undefined }
    },
    onExit: (listener) => {
      exitListener = listener
      return { dispose: () => undefined }
    },
    emitData: (data) => {
      dataListener?.(data)
    },
    emitExit: (event) => {
      exitListener?.(event)
    },
  }
}

describe('TerminalPtyService', () => {
  it('spawns a repo-scoped PTY session with shell, cwd, and terminal env', async () => {
    const ptyProcess = createMockPtyProcess()
    const ptySpawn = vi.fn().mockReturnValue(ptyProcess)

    const service = new TerminalPtyService({
      repoRoot: '/repo/tauri',
      shell: '/bin/zsh',
      ptySpawn,
      environment: { PATH: '/usr/bin' },
    })

    await expect(service.connect(120, 32)).resolves.toMatchObject({
      status: 'ready',
      cwd: '/repo/tauri',
      shell: '/bin/zsh',
      cols: 120,
      rows: 32,
    })

    expect(ptySpawn).toHaveBeenCalledWith(
      '/bin/zsh',
      [],
      expect.objectContaining({
        name: 'xterm-256color',
        cols: 120,
        rows: 32,
        cwd: '/repo/tauri',
        env: expect.objectContaining({
          PATH: '/usr/bin',
          TERM: 'xterm-256color',
          COLORTERM: 'truecolor',
        }),
      }),
    )
  })

  it('forwards writes, resizes, and PTY output after connect', async () => {
    const ptyProcess = createMockPtyProcess()
    const service = new TerminalPtyService({
      repoRoot: '/repo/tauri',
      shell: '/bin/zsh',
      ptySpawn: vi.fn().mockReturnValue(ptyProcess),
    })
    const onData = vi.fn()
    service.onData(onData)

    await service.connect(100, 28)
    await service.write('ls\n')
    await service.resize(140, 36)
    ptyProcess.emitData('file-one\r\n')

    expect(ptyProcess.write).toHaveBeenCalledWith('ls\n')
    expect(ptyProcess.resize).toHaveBeenCalledWith(140, 36)
    expect(onData).toHaveBeenCalledWith('file-one\r\n')
  })

  it('restarts by killing the current PTY and creating a fresh session', async () => {
    const first = createMockPtyProcess()
    const second = createMockPtyProcess()
    const ptySpawn = vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second)

    const service = new TerminalPtyService({
      repoRoot: '/repo/tauri',
      shell: '/bin/zsh',
      ptySpawn,
    })

    await service.connect(90, 24)
    await service.restart(120, 40)

    expect(first.kill).toHaveBeenCalledTimes(1)
    expect(ptySpawn).toHaveBeenNthCalledWith(2, '/bin/zsh', [], expect.objectContaining({ cols: 120, rows: 40 }))
  })

  it('publishes deterministic state and output in mock mode', async () => {
    const service = new TerminalPtyService({ repoRoot: '/repo/tauri', mockMode: true, shell: '/bin/zsh' })
    const onData = vi.fn()
    const onState = vi.fn()

    service.onData(onData)
    service.subscribe(onState)

    await expect(service.connect(80, 24)).resolves.toMatchObject({
      status: 'ready',
      sessionId: 'mock-terminal-session',
      shell: '/bin/zsh',
      cwd: '/repo/tauri',
    })

    expect(onData).toHaveBeenCalledWith(expect.stringContaining('Mock terminal session ready'))
    expect(onState).toHaveBeenCalledWith(
      expect.objectContaining({
        status: 'ready',
        cwd: '/repo/tauri',
      }),
    )
    expect(service.getState()).toEqual(
      expect.objectContaining({
        status: 'ready',
        cwd: '/repo/tauri',
        shell: '/bin/zsh',
        cols: 80,
        rows: 24,
        sessionId: 'mock-terminal-session',
      }),
    )
  })
})
