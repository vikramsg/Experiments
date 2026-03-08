import type { ReactNode } from 'react'

type AppShellProps = {
  children: ReactNode
}

export function AppShell({ children }: AppShellProps) {
  return (
    <main className="workspace-shell">
      <aside className="workspace-nav" aria-label="Apps">
        <p className="workspace-kicker">Workspace</p>
        <h1 className="workspace-title">Hello World</h1>
        <nav>
          <button className="workspace-app workspace-app--active" type="button" aria-current="page">
            <span className="workspace-app-label">Text Editor</span>
            <span className="workspace-app-meta">First app view</span>
          </button>
        </nav>
      </aside>

      <section className="workspace-main">{children}</section>
    </main>
  )
}
