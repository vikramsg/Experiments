import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { WorkspaceApi } from '../../../workspace-contract'
import type { WorkspaceSnapshot } from '../../../workspace-model'
import { App } from './App'

function createApi(snapshot: WorkspaceSnapshot): WorkspaceApi {
  const api: WorkspaceApi = {
    loadState: vi.fn().mockResolvedValue(snapshot),
    saveNotes: vi.fn().mockResolvedValue(undefined),
    adjustSplitter: vi.fn().mockResolvedValue(undefined),
    onStateChange: vi.fn().mockReturnValue(() => undefined),
  }

  return api
}

describe('Notes App', () => {
  it('loads and renders saved notes without browser controls', async () => {
    const api = createApi({
      notes: 'Saved note',
      notesWidth: 420,
      browserUrl: 'https://example.com',
      canGoBack: false,
      canGoForward: false,
    })

    render(<App api={api} />)

    expect(await screen.findByRole('textbox', { name: /notes editor/i })).toHaveValue('Saved note')
    expect(screen.queryByRole('textbox', { name: /browser url/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^go$/i })).not.toBeInTheDocument()
  })

  it('clears the loading state when workspace updates arrive after an initial load failure', async () => {
    const api = createApi({
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
      canGoBack: false,
      canGoForward: false,
    })
    let listener: (snapshot: WorkspaceSnapshot) => void = () => undefined

    vi.mocked(api.loadState).mockRejectedValue(new Error('Workspace is not open'))
    vi.mocked(api.onStateChange).mockImplementation((nextListener) => {
      listener = nextListener
      return () => undefined
    })

    render(<App api={api} />)

    expect(screen.getByText(/loading workspace/i)).toBeVisible()

    act(() => {
      listener({
        notes: 'Recovered note',
        notesWidth: 420,
        browserUrl: 'https://example.com',
        canGoBack: false,
        canGoForward: false,
      })
    })

    await waitFor(() => {
      expect(screen.getByRole('textbox', { name: /notes editor/i })).toHaveValue('Recovered note')
      expect(screen.getByText(/auto-saving notes to your workspace/i)).toBeVisible()
      expect(screen.queryByText(/loading workspace/i)).not.toBeInTheDocument()
    })
  })

  it('saves note changes as the user types', async () => {
    const user = userEvent.setup()
    const api = createApi({
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
      canGoBack: false,
      canGoForward: false,
    })

    render(<App api={api} />)
    const editor = await screen.findByRole('textbox', { name: /notes editor/i })
    await user.type(editor, 'Draft')

    await waitFor(() => {
      expect(api.saveNotes).toHaveBeenLastCalledWith('Draft')
    })
  })
})
