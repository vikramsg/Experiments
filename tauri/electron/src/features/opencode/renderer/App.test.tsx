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
    expect(screen.getByText(/local opencode server behind a narrow electron bridge/i)).toBeVisible()
    expect(screen.getByText(/repo scope/i)).toBeVisible()
    expect(screen.getByDisplayValue('')).toBeVisible()
  })

  it('submits prompts through the renderer api', async () => {
    const user = userEvent.setup()
    const api = createApi(createDefaultOpenCodeState('/repo/tauri'))

    render(<App api={api} />)

    await user.type(await screen.findByRole('textbox', { name: /ask opencode/i }), 'Summarize this repo')
    await user.click(screen.getByRole('button', { name: /send prompt/i }))

    expect(api.sendPrompt).toHaveBeenCalledWith('Summarize this repo')
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
