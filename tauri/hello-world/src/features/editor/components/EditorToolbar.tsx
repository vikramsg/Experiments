type EditorToolbarProps = {
  onBackToApps?: () => void
  onNew: () => void
  onOpen: () => void
  onSave: () => void
  onSaveAs: () => void
  isSaving: boolean
}

export function EditorToolbar({ onBackToApps, onNew, onOpen, onSave, onSaveAs, isSaving }: EditorToolbarProps) {
  return (
    <div className="editor-toolbar">
      <div className="editor-heading">
        {onBackToApps ? (
          <button className="editor-back" type="button" onClick={onBackToApps}>
            Back to Apps
          </button>
        ) : null}
        <div>
          <p className="editor-kicker">App View</p>
          <h2>Text Editor</h2>
        </div>
      </div>

      <div className="editor-actions">
        <button type="button" onClick={onNew}>
          New
        </button>
        <button type="button" onClick={onOpen}>
          Open
        </button>
        <button type="button" onClick={onSave} disabled={isSaving}>
          {isSaving ? 'Saving...' : 'Save'}
        </button>
        <button type="button" onClick={onSaveAs} disabled={isSaving}>
          Save As
        </button>
      </div>
    </div>
  )
}
