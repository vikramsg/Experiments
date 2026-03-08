export type OpenedTextFile = {
  path: string
  name: string
  content: string
}

export type FileService = {
  openTextFile: () => Promise<OpenedTextFile | null>
  saveTextFile: (path: string, content: string) => Promise<void>
  saveTextFileAs: (suggestedName: string, content: string) => Promise<string | null>
}
