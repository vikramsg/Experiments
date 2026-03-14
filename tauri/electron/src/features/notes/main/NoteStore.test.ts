import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { NoteStore } from './NoteStore'

describe('NoteStore', () => {
  let userDataPath: string

  beforeEach(async () => {
    userDataPath = await mkdtemp(join(tmpdir(), 'electron-note-store-'))
  })

  afterEach(async () => {
    await rm(userDataPath, { recursive: true, force: true })
  })

  it('returns defaults before anything is saved', async () => {
    const store = new NoteStore(userDataPath)

    await expect(store.load()).resolves.toEqual({
      notes: '',
      notesWidth: 420,
      browserUrl: 'https://example.com',
    })
  })

  it('persists notes, splitter width, and browser url', async () => {
    const store = new NoteStore(userDataPath)

    await store.save({
      notes: 'Saved from test',
      notesWidth: 512,
      browserUrl: 'https://example.org/docs',
    })

    await expect(store.load()).resolves.toEqual({
      notes: 'Saved from test',
      notesWidth: 512,
      browserUrl: 'https://example.org/docs',
    })

    await expect(readFile(join(userDataPath, 'workspace-state.json'), 'utf8')).resolves.toContain('Saved from test')
  })
})
