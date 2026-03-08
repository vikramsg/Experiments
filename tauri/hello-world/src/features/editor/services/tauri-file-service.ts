import { open, save } from '@tauri-apps/plugin-dialog'
import { readTextFile, writeTextFile } from '@tauri-apps/plugin-fs'

import type { FileService, OpenedTextFile } from './file-service'

function getFileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? 'Untitled'
}

export async function openTextFile(): Promise<OpenedTextFile | null> {
  const selected = await open({
    title: 'Open text file',
    multiple: false,
    directory: false,
    filters: [{ name: 'Text', extensions: ['txt', 'md', 'markdown'] }],
  })

  if (!selected || Array.isArray(selected)) {
    return null
  }

  const content = await readTextFile(selected)

  return {
    path: selected,
    name: getFileName(selected),
    content,
  }
}

export async function saveTextFile(path: string, content: string): Promise<void> {
  await writeTextFile(path, content)
}

export async function saveTextFileAs(suggestedName: string, content: string): Promise<string | null> {
  const selected = await save({
    title: 'Save text file',
    defaultPath: suggestedName,
    filters: [{ name: 'Text', extensions: ['txt', 'md', 'markdown'] }],
  })

  if (!selected) {
    return null
  }

  await writeTextFile(selected, content)

  return selected
}

export const tauriFileService: FileService = {
  openTextFile,
  saveTextFile,
  saveTextFileAs,
}
