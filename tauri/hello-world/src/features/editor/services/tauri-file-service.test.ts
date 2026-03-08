import { beforeEach, describe, expect, it, vi } from 'vitest'

const { openMock, saveMock, readTextFileMock, writeTextFileMock } = vi.hoisted(() => ({
  openMock: vi.fn(),
  saveMock: vi.fn(),
  readTextFileMock: vi.fn(),
  writeTextFileMock: vi.fn(),
}))

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: openMock,
  save: saveMock,
}))

vi.mock('@tauri-apps/plugin-fs', () => ({
  readTextFile: readTextFileMock,
  writeTextFile: writeTextFileMock,
}))

import { openTextFile, saveTextFile, saveTextFileAs } from './tauri-file-service'

describe('tauri file service', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('opens a file through the native dialog and reads its text', async () => {
    openMock.mockResolvedValue('/tmp/example.md')
    readTextFileMock.mockResolvedValue('# Example')

    await expect(openTextFile()).resolves.toEqual({
      path: '/tmp/example.md',
      name: 'example.md',
      content: '# Example',
    })
    expect(openMock).toHaveBeenCalled()
    expect(readTextFileMock).toHaveBeenCalledWith('/tmp/example.md')
  })

  it('writes the current document to an existing path', async () => {
    await saveTextFile('/tmp/example.md', 'Updated body')

    expect(writeTextFileMock).toHaveBeenCalledWith('/tmp/example.md', 'Updated body')
  })

  it('uses save as to choose a new path before writing', async () => {
    saveMock.mockResolvedValue('/tmp/new-note.md')

    await expect(saveTextFileAs('Untitled.md', 'Fresh note')).resolves.toBe('/tmp/new-note.md')
    expect(saveMock).toHaveBeenCalled()
    expect(writeTextFileMock).toHaveBeenCalledWith('/tmp/new-note.md', 'Fresh note')
  })
})
