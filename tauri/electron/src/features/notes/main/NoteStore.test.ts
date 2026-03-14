import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { DEFAULT_WORKSPACE_SNAPSHOT, type PersistedWorkspaceSnapshot } from '../../../workspace-model'

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

    await expect(store.load()).resolves.toEqual(DEFAULT_WORKSPACE_SNAPSHOT)
  })

  it('persists notes, splitter width, and browser url without storing history flags', async () => {
    const store = new NoteStore(userDataPath)

    await store.save({
      notes: 'Saved from test',
      notesWidth: 512,
      browserUrl: 'https://example.org/docs',
      canGoBack: true,
      canGoForward: true,
    })

    await expect(store.load()).resolves.toEqual({
      notes: 'Saved from test',
      notesWidth: 512,
      browserUrl: 'https://example.org/docs',
      canGoBack: false,
      canGoForward: false,
    })

    await expect(readFile(join(userDataPath, 'workspace-state.json'), 'utf8')).resolves.toContain('Saved from test')
    await expect(readFile(join(userDataPath, 'workspace-state.json'), 'utf8')).resolves.not.toContain('canGoBack')
    await expect(readFile(join(userDataPath, 'workspace-state.json'), 'utf8').then((raw) => JSON.parse(raw) as PersistedWorkspaceSnapshot)).resolves.toEqual({
      notes: 'Saved from test',
      notesWidth: 512,
      browserUrl: 'https://example.org/docs',
    })
  })
})
