type AppSelectorProps = {
  onLaunchTextEditor: () => void
}

export function AppSelector({ onLaunchTextEditor }: AppSelectorProps) {
  return (
    <section className="app-selector" aria-labelledby="app-selector-title">
      <div className="app-selector__hero">
        <p className="workspace-kicker">Hello World</p>
        <h1 id="app-selector-title" className="workspace-title">
          Choose an App
        </h1>
        <p className="workspace-lede">
          Start in the launcher, then move into a focused app experience. The first app is a desktop text editor.
        </p>
      </div>

      <div className="app-grid" role="list" aria-label="Available apps">
        <article className="app-card" role="listitem">
          <p className="app-card__eyebrow">Ready now</p>
          <h2>Text Editor</h2>
          <p>
            Open local files, write notes, edit markdown, and save changes back to disk through native Tauri dialogs.
          </p>
          <button className="app-card__button" type="button" onClick={onLaunchTextEditor}>
            Launch Text Editor
          </button>
        </article>
      </div>
    </section>
  )
}
