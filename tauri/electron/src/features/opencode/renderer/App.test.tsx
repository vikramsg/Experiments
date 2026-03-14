import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { OpenCodeApi } from '../../../opencode-contract'
import type { OpenCodeState } from '../../../opencode-model'
import { createDefaultOpenCodeState } from '../../../opencode-model'
import { App } from './App'

function createApi(state: OpenCodeState): OpenCodeApi {
  return {
    loadState: vi.fn().mockResolvedValue(state),
    sendPrompt: vi.fn().mockResolvedValue(undefined),
    onStateChange: vi.fn().mockReturnValue(() => undefined),
  }
}

describe('OpenCode App', () => {
  it('loads the initial repo-scoped state', async () => {
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))

    render(<App api={api} />)

    expect(await screen.findByRole('heading', { name: /opencode/i })).toBeVisible()
    expect(screen.getByText(/local opencode server beside a live browser surface/i)).toBeVisible()
    expect(screen.getByText(/repo scope/i)).toBeVisible()
    expect(screen.getByDisplayValue('')).toBeVisible()
    expect(screen.getAllByText(/ask what opencode sees in the browser/i).length).toBeGreaterThan(0)
  })

  it('submits prompts through the renderer api', async () => {
    const user = userEvent.setup()
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))

    render(<App api={api} />)

    await user.type(await screen.findByRole('textbox', { name: /ask opencode/i }), 'Summarize this repo')
    await user.click(screen.getByRole('button', { name: /send prompt/i }))

    expect(api.sendPrompt).toHaveBeenCalledWith('Summarize this repo')
  })

  it('submits the prompt when Enter is pressed without Shift', async () => {
    const user = userEvent.setup()
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))

    render(<App api={api} />)

    const prompt = await screen.findByRole('textbox', { name: /ask opencode/i })
    await user.type(prompt, 'Summarize this repo')
    await user.keyboard('{Enter}')

    expect(api.sendPrompt).toHaveBeenCalledWith('Summarize this repo')
  })

  it('keeps a newline and does not submit when Shift+Enter is pressed', async () => {
    const user = userEvent.setup()
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))

    render(<App api={api} />)

    const prompt = await screen.findByRole('textbox', { name: /ask opencode/i })
    await user.type(prompt, 'Line one')
    await user.keyboard('{Shift>}{Enter}{/Shift}')

    expect(api.sendPrompt).not.toHaveBeenCalled()
    expect(prompt).toHaveValue('Line one\n')
  })

  it('ignores empty trimmed prompts', async () => {
    const user = userEvent.setup()
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))

    render(<App api={api} />)

    const prompt = await screen.findByRole('textbox', { name: /ask opencode/i })
    await user.type(prompt, '   ')
    await user.keyboard('{Enter}')

    expect(api.sendPrompt).not.toHaveBeenCalled()
  })

  it('uses a fixed-height shell with a compact send button', async () => {
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))

    render(<App api={api} />)

    const heading = await screen.findByRole('heading', { name: /opencode/i })
    const page = heading.closest('main')
    const sendButton = screen.getByRole('button', { name: /send prompt/i })

    expect(page).toHaveStyle({ height: '100vh', overflow: 'hidden' })
    expect(sendButton).toHaveStyle({ height: '52px' })
  })

  it('reacts to state updates and renders assistant replies', async () => {
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))
    let listener: ((state: OpenCodeState) => void) | undefined

    vi.mocked(api.onStateChange).mockImplementation((nextListener) => {
      listener = nextListener
      return () => undefined
    })

    render(<App api={api} />)

    await screen.findByRole('heading', { name: /opencode/i })

    act(() => {
      listener?.({
        status: 'ready',
        repoRoot: '/repo/tauri',
        sessionId: 'session-1',
        error: null,
        messages: [
          ...createDefaultOpenCodeState('/repo/tauri').messages,
          { id: 'u1', role: 'user', text: 'What is this app?' },
          { id: 'a1', role: 'assistant', text: 'It is a read-only repo chat app built on OpenCode.' },
        ],
      })
    })

    await waitFor(() => {
      expect(screen.getByText(/what is this app\?/i)).toBeVisible()
      expect(screen.getByText(/read-only repo chat app built on opencode/i)).toBeVisible()
    })
  })
})
