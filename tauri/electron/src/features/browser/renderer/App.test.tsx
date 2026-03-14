import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { WorkspaceSnapshot } from '../../../shared/types/workspace'
import type { WorkspaceApi } from '../../../types'
import { App } from './App'

function createApi(snapshot: WorkspaceSnapshot): WorkspaceApi {
  return {
    loadState: vi.fn().mockResolvedValue(snapshot),
    saveNotes: vi.fn().mockResolvedValue(undefined),
    setBrowserUrl: vi.fn().mockResolvedValue(undefined),
    adjustSplitter: vi.fn().mockResolvedValue(undefined),
    onStateChange: vi.fn().mockReturnValue(() => undefined),
  }
}

describe('Browser Chrome App', () => {
  it('loads the saved browser url and navigates on go', async () => {
    const user = userEvent.setup()
    const api = createApi({
      notes: 'Saved note',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    render(<App api={api} />)

    const input = await screen.findByRole('textbox', { name: /browser url/i })
    expect(input).toHaveValue('https://example.com')

    await user.clear(input)
    await user.type(input, 'https://example.org/docs')
    await user.click(screen.getByRole('button', { name: /^go$/i }))

    expect(api.setBrowserUrl).toHaveBeenCalledWith('https://example.org/docs')
  })

  it('reacts to workspace state updates', async () => {
    const api = createApi({
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    let listener: ((snapshot: WorkspaceSnapshot) => void) | undefined
    vi.mocked(api.onStateChange).mockImplementation((nextListener) => {
      listener = nextListener
      return () => undefined
    })

    render(<App api={api} />)

    const input = await screen.findByRole('textbox', { name: /browser url/i })

    act(() => {
      listener?.({ notes: '', notesWidth: 420, browserUrl: 'https://example.net/app' })
    })

    expect(input).toHaveValue('https://example.net/app')
  })
})
