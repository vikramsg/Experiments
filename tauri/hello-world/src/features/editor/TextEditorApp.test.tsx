import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { FileService } from './services/file-service'

vi.mock('./components/CodeEditor', () => ({
  CodeEditor: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <textarea aria-label="Editor" value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}))

import { TextEditorApp } from './TextEditorApp'

describe('TextEditorApp', () => {
  let fileService: FileService

  beforeEach(() => {
    fileService = {
      openTextFile: vi.fn(),
      saveTextFile: vi.fn(),
      saveTextFileAs: vi.fn(),
    }
  })

  it('starts with an untitled clean document and becomes dirty after editing', () => {
    render(<TextEditorApp fileService={fileService} />)

    expect(screen.getAllByText('Untitled').length).toBeGreaterThan(0)
    expect(screen.getAllByText('All changes saved').length).toBeGreaterThan(0)

    fireEvent.change(screen.getByRole('textbox', { name: /editor/i }), { target: { value: 'Draft note' } })

    expect(screen.getAllByText('Unsaved changes')).toHaveLength(2)
  })

  it('opens a file and shows its content', async () => {
    vi.mocked(fileService.openTextFile).mockResolvedValue({
      path: '/tmp/day-one.md',
      name: 'day-one.md',
      content: '# Day One',
    })

    render(<TextEditorApp fileService={fileService} />)
    fireEvent.click(screen.getByRole('button', { name: /open/i }))

    await waitFor(() => {
      expect(screen.getByDisplayValue('# Day One')).toBeInTheDocument()
    })
    expect(screen.getAllByText('day-one.md').length).toBeGreaterThan(0)
  })

  it('saves the current document to a new path and clears dirty state', async () => {
    vi.mocked(fileService.saveTextFileAs).mockResolvedValue('/tmp/note.md')

    render(<TextEditorApp fileService={fileService} />)

    fireEvent.change(screen.getByRole('textbox', { name: /editor/i }), { target: { value: 'Hello world' } })
    fireEvent.click(screen.getByRole('button', { name: /^save$/i }))

    await waitFor(() => {
      expect(fileService.saveTextFileAs).toHaveBeenCalledWith('Untitled.md', 'Hello world')
    })
    expect(screen.getAllByText('note.md').length).toBeGreaterThan(0)
    expect(screen.getAllByText('All changes saved')).toHaveLength(2)
  })
})
