import { createDocument, markSaved, updateContent } from './editor-document'

describe('editor document model', () => {
  it('creates an untitled clean document by default', () => {
    expect(createDocument()).toEqual({
      content: '',
      filePath: null,
      fileName: 'Untitled',
      isDirty: false,
    })
  })

  it('marks the document dirty after content changes', () => {
    const next = updateContent(createDocument(), 'Hello world')

    expect(next.content).toBe('Hello world')
    expect(next.isDirty).toBe(true)
  })

  it('clears dirty state and tracks file metadata after save', () => {
    const saved = markSaved(updateContent(createDocument(), 'Draft'), '/tmp/note.md')

    expect(saved.filePath).toBe('/tmp/note.md')
    expect(saved.fileName).toBe('note.md')
    expect(saved.isDirty).toBe(false)
  })
})
