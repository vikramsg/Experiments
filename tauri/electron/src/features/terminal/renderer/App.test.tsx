import { act, render, screen, waitFor } from '@testing-library/react'

import type { TerminalApi } from '../../../terminal-contract'
import { createDefaultTerminalState, type TerminalState } from '../../../terminal-model'
import { App } from './App'
import { createGhosttyRuntime } from './ghostty-runtime'

vi.mock('./ghostty-runtime', () => ({
  createGhosttyRuntime: vi.fn(),
}))

function createApi(state: TerminalState): TerminalApi {
  return {
    loadState: vi.fn().mockResolvedValue(state),
    connect: vi.fn().mockResolvedValue(undefined),
    write: vi.fn().mockResolvedValue(undefined),
    resize: vi.fn().mockResolvedValue(undefined),
    restart: vi.fn().mockResolvedValue(undefined),
    onData: vi.fn().mockReturnValue(() => undefined),
    onStateChange: vi.fn().mockReturnValue(() => undefined),
  }
}

describe('Terminal App', () => {
  it('loads a minimal terminal-first surface and connects on mount', async () => {
    const api = createApi(createDefaultTerminalState('/repo/tauri', '/bin/zsh'))
    vi.mocked(createGhosttyRuntime).mockResolvedValue({
      write: vi.fn(),
      focus: vi.fn(),
      dispose: vi.fn(),
      getSize: () => ({ cols: 120, rows: 32 }),
    })

    render(<App api={api} />)

    expect(await screen.findByLabelText(/terminal surface/i)).toBeVisible()
    expect(screen.queryByRole('heading', { name: /terminal/i })).not.toBeInTheDocument()
    expect(screen.queryByText(/full local shell access through a main-process pty/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /restart terminal/i })).not.toBeInTheDocument()
    await waitFor(() => {
      expect(api.connect).toHaveBeenCalledTimes(1)
      expect(api.connect).toHaveBeenCalledWith(expect.any(Number), expect.any(Number))
    })
  })

  it('exposes terminal state through a hidden status node instead of visible chrome', async () => {
    const api = createApi(createDefaultTerminalState('/repo/tauri', '/bin/zsh'))
    vi.mocked(createGhosttyRuntime).mockResolvedValue({
      write: vi.fn(),
      focus: vi.fn(),
      dispose: vi.fn(),
      getSize: () => ({ cols: 120, rows: 32 }),
    })

    render(<App api={api} />)

    expect(await screen.findByTestId('terminal-status')).toHaveTextContent(/preparing|starting|local shell/i)
  })

  it('writes streamed terminal data into the ghostty runtime', async () => {
    const api = createApi(createDefaultTerminalState('/repo/tauri', '/bin/zsh'))
    let onData: ((data: string) => void) | undefined
    const runtime = {
      write: vi.fn(),
      focus: vi.fn(),
      dispose: vi.fn(),
      getSize: () => ({ cols: 120, rows: 32 }),
    }

    vi.mocked(api.onData).mockImplementation((listener) => {
      onData = listener
      return () => undefined
    })
    vi.mocked(createGhosttyRuntime).mockResolvedValue(runtime)

    render(<App api={api} />)
    await screen.findByLabelText(/terminal surface/i)
    await waitFor(() => {
      expect(api.connect).toHaveBeenCalledTimes(1)
    })

    act(() => {
      onData?.('hello from shell\r\n')
    })

    expect(runtime.write).toHaveBeenCalledWith('hello from shell\r\n')
  })

  it('reacts to state updates without rendering shell metadata cards', async () => {
    const api = createApi(createDefaultTerminalState('/repo/tauri', '/bin/zsh'))
    let onStateChange: ((state: TerminalState) => void) | undefined

    vi.mocked(api.onStateChange).mockImplementation((listener) => {
      onStateChange = listener
      return () => undefined
    })
    vi.mocked(createGhosttyRuntime).mockResolvedValue({
      write: vi.fn(),
      focus: vi.fn(),
      dispose: vi.fn(),
      getSize: () => ({ cols: 120, rows: 32 }),
    })

    render(<App api={api} />)
    await screen.findByLabelText(/terminal surface/i)

    act(() => {
      onStateChange?.({
        ...createDefaultTerminalState('/repo/tauri/electron', '/bin/zsh'),
        status: 'ready',
        cwd: '/repo/tauri/electron',
        shell: '/bin/zsh',
        cols: 120,
        rows: 32,
        sessionId: 'session-1',
        error: null,
        exitCode: null,
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId('terminal-status')).toHaveTextContent(/local shell is ready/i)
    })
    expect(screen.queryByText('/repo/tauri/electron')).not.toBeInTheDocument()
    expect(screen.queryByText('/bin/zsh')).not.toBeInTheDocument()
  })
})
