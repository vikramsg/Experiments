import { randomUUID } from 'node:crypto'

import { createDefaultTerminalState, type TerminalAppearance, type TerminalState } from '../../../terminal-model'

type PtySpawnOptions = {
  name?: string
  cols?: number
  rows?: number
  cwd?: string
  env?: Record<string, string | undefined>
}

type PtyDisposable = { dispose: () => void }

type PtyProcess = {
  write: (data: string) => void
  resize: (cols: number, rows: number) => void
  kill: () => void
  onData: (listener: (data: string) => void) => PtyDisposable
  onExit: (listener: (event: { exitCode: number; signal?: number }) => void) => PtyDisposable
}

type PtySpawnLike = (file: string, args: string[], options: PtySpawnOptions) => PtyProcess

type Disposable = { dispose: () => void }

async function loadNodePtySpawn(): Promise<PtySpawnLike> {
  const nodePtyModule = (await import('@lydell/node-pty')) as { spawn: PtySpawnLike }
  return nodePtyModule.spawn
}

export function resolveTerminalShell(environment: NodeJS.ProcessEnv = process.env, preferredShell?: string): string {
  if (preferredShell) {
    return preferredShell
  }

  if (process.platform === 'win32') {
    return environment.COMSPEC || 'cmd.exe'
  }

  return environment.SHELL || '/bin/bash'
}

export class TerminalPtyService {
  private readonly stateListeners = new Set<(state: TerminalState) => void>()
  private readonly dataListeners = new Set<(data: string) => void>()
  private readonly environment: NodeJS.ProcessEnv
  private readonly shell: string
  private state: TerminalState
  private ptyProcess: PtyProcess | null = null
  private ptyDisposables: Disposable[] = []

  constructor(
    private readonly input: {
      repoRoot: string
      appearance?: TerminalAppearance
      shell?: string
      mockMode?: boolean
      ptySpawn?: PtySpawnLike
      environment?: NodeJS.ProcessEnv
    },
  ) {
    this.environment = {
      ...process.env,
      ...(input.environment ?? {}),
    }
    this.shell = resolveTerminalShell(this.environment, input.shell)
    this.state = createDefaultTerminalState(input.repoRoot, this.shell, input.appearance)
  }

  getState(): TerminalState {
    return this.state
  }

  subscribe(listener: (state: TerminalState) => void): () => void {
    this.stateListeners.add(listener)
    return () => {
      this.stateListeners.delete(listener)
    }
  }

  onData(listener: (data: string) => void): () => void {
    this.dataListeners.add(listener)
    return () => {
      this.dataListeners.delete(listener)
    }
  }

  async connect(cols: number, rows: number): Promise<TerminalState> {
    if (this.ptyProcess) {
      await this.resize(cols, rows)
      return this.state
    }

    this.updateState({
      status: 'connecting',
      cols,
      rows,
      error: null,
      exitCode: null,
      cwd: this.input.repoRoot,
      shell: this.shell,
    })

    if (this.useMockMode()) {
      this.updateState({
        status: 'ready',
        sessionId: 'mock-terminal-session',
        error: null,
        exitCode: null,
      })
      this.publishData(
        `Mock terminal session ready in ${this.input.repoRoot}\r\nShell: ${this.shell}\r\nType commands to simulate a local terminal.\r\n$ `,
      )
      return this.state
    }

    try {
      const ptySpawn = this.input.ptySpawn ?? (await loadNodePtySpawn())
      const ptyProcess = ptySpawn(this.shell, [], {
        name: 'xterm-256color',
        cols,
        rows,
        cwd: this.input.repoRoot,
        env: {
          ...this.environment,
          TERM: 'xterm-256color',
          COLORTERM: 'truecolor',
        },
      })

      const onDataDisposable = ptyProcess.onData((data) => {
        this.publishData(data)
      })
      const onExitDisposable = ptyProcess.onExit(({ exitCode }) => {
        this.ptyProcess = null
        this.disposeProcessListeners()
        this.updateState({
          status: 'exited',
          sessionId: null,
          exitCode,
          error: null,
        })
      })

      this.ptyProcess = ptyProcess
      this.ptyDisposables = [onDataDisposable, onExitDisposable]

      this.updateState({
        status: 'ready',
        sessionId: randomUUID(),
        error: null,
        exitCode: null,
      })

      return this.state
    } catch (error) {
      this.updateState({
        status: 'error',
        error: error instanceof Error ? error.message : 'Failed to start the local shell.',
      })
      throw error
    }
  }

  async write(data: string): Promise<void> {
    if (!data) {
      return
    }

    if (this.useMockMode()) {
      this.publishMockCommand(data)
      return
    }

    this.ptyProcess?.write(data)
  }

  async resize(cols: number, rows: number): Promise<void> {
    this.updateState({ cols, rows })

    if (this.useMockMode()) {
      return
    }

    this.ptyProcess?.resize(cols, rows)
  }

  async restart(cols: number, rows: number): Promise<TerminalState> {
    this.teardownCurrentProcess(true)
    this.updateState({ status: 'idle', sessionId: null, exitCode: null, error: null })
    return this.connect(cols, rows)
  }

  async dispose(): Promise<void> {
    this.teardownCurrentProcess(true)
  }

  private useMockMode(): boolean {
    return this.input.mockMode ?? process.env.ELECTRON_TERMINAL_MOCK === '1'
  }

  private publishMockCommand(data: string) {
    const command = data.replace(/\r/g, '').trim()
    if (!command) {
      return
    }

    const response = this.buildMockResponse(command)
    this.publishData(`${command}\r\n${response}\r\n$ `)
  }

  private buildMockResponse(command: string): string {
    if (command === 'pwd') {
      return this.input.repoRoot
    }

    if (command === 'echo $SHELL') {
      return this.shell
    }

    if (command === 'tmux -V') {
      return 'tmux 3.5a (mock)'
    }

    if (command === 'ls') {
      return 'README.md\r\nsrc\r\ndocs'
    }

    return `Mock command executed: ${command}`
  }

  private publishData(data: string) {
    for (const listener of this.dataListeners) {
      listener(data)
    }
  }

  private updateState(partial: Partial<TerminalState>) {
    this.state = {
      ...this.state,
      ...partial,
    }

    for (const listener of this.stateListeners) {
      listener(this.state)
    }
  }

  private disposeProcessListeners() {
    for (const disposable of this.ptyDisposables) {
      disposable.dispose()
    }
    this.ptyDisposables = []
  }

  private teardownCurrentProcess(kill: boolean) {
    const process = this.ptyProcess
    this.ptyProcess = null
    this.disposeProcessListeners()

    if (kill) {
      process?.kill()
    }
  }
}
