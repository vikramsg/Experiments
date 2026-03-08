export type EditorDocument = {
  content: string
  filePath: string | null
  fileName: string
  isDirty: boolean
}

export type OpenedTextFile = {
  path: string
  name: string
  content: string
}

export function createDocument(): EditorDocument {
  return {
    content: '',
    filePath: null,
    fileName: 'Untitled',
    isDirty: false,
  }
}

export function createDocumentFromFile(file: OpenedTextFile): EditorDocument {
  return {
    content: file.content,
    filePath: file.path,
    fileName: file.name,
    isDirty: false,
  }
}

export function updateContent(document: EditorDocument, content: string): EditorDocument {
  if (content === document.content) {
    return document
  }

  return {
    ...document,
    content,
    isDirty: true,
  }
}

export function markSaved(document: EditorDocument, filePath: string): EditorDocument {
  return {
    ...document,
    filePath,
    fileName: filePath.split(/[\\/]/).pop() ?? 'Untitled',
    isDirty: false,
  }
}
