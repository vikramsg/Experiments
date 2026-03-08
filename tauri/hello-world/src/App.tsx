import './App.css'

import { AppShell } from './features/app-shell/AppShell'
import { TextEditorApp } from './features/editor/TextEditorApp'

function App() {
  return (
    <AppShell>
      <TextEditorApp />
    </AppShell>
  )
}

export default App
