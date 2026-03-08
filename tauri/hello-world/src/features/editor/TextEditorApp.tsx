import { useEffect, useMemo, useState } from 'react'

import {
  createDocument,
  createDocumentFromFile,
  markSaved,
  updateContent,
  type EditorDocument,
} from './model/editor-document'
import { CodeEditor } from './components/CodeEditor'
import { EditorStatusBar } from './components/EditorStatusBar'
import { EditorToolbar } from './components/EditorToolbar'
import { tauriFileService } from './services/tauri-file-service'
import type { FileService } from './services/file-service'

type TextEditorAppProps = {
  document?: EditorDocument
  onDocumentChange?: (document: EditorDocument) => void
  onBackToApps?: () => void
  fileService?: FileService
}

export function TextEditorApp({ document, onDocumentChange, onBackToApps, fileService = tauriFileService }: TextEditorAppProps) {
  const [internalDocument, setInternalDocument] = useState<EditorDocument>(() => createDocument())
  const [isSaving, setIsSaving] = useState(false)

  const editorDocument = document ?? internalDocument

  function setEditorDocument(next: EditorDocument | ((current: EditorDocument) => EditorDocument)) {
    const resolved = typeof next === 'function' ? next(editorDocument) : next

    if (onDocumentChange) {
      onDocumentChange(resolved)
      return
    }

    setInternalDocument(resolved)
  }

  const title = useMemo(
    () => `${editorDocument.isDirty ? '* ' : ''}${editorDocument.fileName} - Hello World`,
    [editorDocument.fileName, editorDocument.isDirty],
  )

  useEffect(() => {
    window.document.title = title
  }, [title])

  async function handleOpen() {
    const opened = await fileService.openTextFile()

    if (!opened) {
      return
    }

    setEditorDocument(createDocumentFromFile(opened))
  }

  async function handleSave() {
    setIsSaving(true)

    try {
      if (editorDocument.filePath) {
        await fileService.saveTextFile(editorDocument.filePath, editorDocument.content)
        setEditorDocument((current) => markSaved(current, editorDocument.filePath!))
        return
      }

      const savedPath = await fileService.saveTextFileAs(`${editorDocument.fileName}.md`, editorDocument.content)
      if (savedPath) {
        setEditorDocument((current) => markSaved(current, savedPath))
      }
    } finally {
      setIsSaving(false)
    }
  }

  async function handleSaveAs() {
    setIsSaving(true)

    try {
      const savedPath = await fileService.saveTextFileAs(`${editorDocument.fileName}.md`, editorDocument.content)
      if (savedPath) {
        setEditorDocument((current) => markSaved(current, savedPath))
      }
    } finally {
      setIsSaving(false)
    }
  }

  function handleNew() {
    setEditorDocument(createDocument())
  }

  return (
    <section className="editor-app">
      <EditorToolbar
        onBackToApps={onBackToApps}
        onNew={handleNew}
        onOpen={handleOpen}
        onSave={handleSave}
        onSaveAs={handleSaveAs}
        isSaving={isSaving}
      />

      <div className="editor-surface">
        <div className="editor-summary">
          <p className="editor-summary-label">Current Document</p>
          <p className="editor-summary-name">{editorDocument.fileName}</p>
          <p className="editor-summary-state">{editorDocument.isDirty ? 'Unsaved changes' : 'All changes saved'}</p>
        </div>

        <CodeEditor
          value={editorDocument.content}
          onChange={(value) => setEditorDocument((current) => updateContent(current, value))}
        />
      </div>

      <EditorStatusBar
        fileName={editorDocument.fileName}
        filePath={editorDocument.filePath}
        isDirty={editorDocument.isDirty}
      />
    </section>
  )
}
