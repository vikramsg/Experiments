import { render, screen, waitFor } from '@testing-library/react'
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

describe('Notes App', () => {
  it('loads and renders saved notes without browser controls', async () => {
    const api = createApi({
      notes: 'Saved note',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    render(<App api={api} />)

    expect(await screen.findByRole('textbox', { name: /notes editor/i })).toHaveValue('Saved note')
    expect(screen.queryByRole('textbox', { name: /browser url/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^go$/i })).not.toBeInTheDocument()
  })

  it('saves note changes as the user types', async () => {
    const user = userEvent.setup()
    const api = createApi({
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    render(<App api={api} />)
    const editor = await screen.findByRole('textbox', { name: /notes editor/i })
    await user.type(editor, 'Draft')

    await waitFor(() => {
      expect(api.saveNotes).toHaveBeenLastCalledWith('Draft')
    })
  })
})
