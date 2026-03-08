type EditorStatusBarProps = {
  fileName: string
  filePath: string | null
  isDirty: boolean
}

export function EditorStatusBar({ fileName, filePath, isDirty }: EditorStatusBarProps) {
  return (
    <footer className="editor-statusbar">
      <div>
        <strong>{fileName}</strong>
        <span>{filePath ?? 'No file chosen yet'}</span>
      </div>
      <p>{isDirty ? 'Unsaved changes' : 'All changes saved'}</p>
    </footer>
  )
}
