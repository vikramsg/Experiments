import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { WorkspaceSnapshot } from '../main/note-store'
import type { WorkspaceApi } from '../types'
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
  it('loads and renders saved notes and browser url', async () => {
    const api = createApi({
      notes: 'Saved note',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    render(<App api={api} />)

    expect(await screen.findByRole('textbox', { name: /notes editor/i })).toHaveValue('Saved note')
    expect(screen.getByRole('textbox', { name: /browser url/i })).toHaveValue('https://example.com')
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

  it('navigates the browser when the go button is pressed', async () => {
    const user = userEvent.setup()
    const api = createApi({
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })

    render(<App api={api} />)
    const urlInput = await screen.findByRole('textbox', { name: /browser url/i })
    await user.clear(urlInput)
    await user.type(urlInput, 'https://example.org/docs')
    await user.click(screen.getByRole('button', { name: /^go$/i }))

    expect(api.setBrowserUrl).toHaveBeenCalledWith('https://example.org/docs')
  })
})
