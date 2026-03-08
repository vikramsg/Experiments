import './App.css'
import { useEffect, useState } from 'react'

import { AppShell } from './features/app-shell/AppShell'
import { AppSelector } from './features/app-shell/AppSelector'
import { TextEditorApp } from './features/editor/TextEditorApp'
import { createDocument, type EditorDocument } from './features/editor/model/editor-document'

type AppView = 'selector' | 'text-editor'

function App() {
  const [currentView, setCurrentView] = useState<AppView>('selector')
  const [textEditorDocument, setTextEditorDocument] = useState<EditorDocument>(() => createDocument())

  useEffect(() => {
    if (currentView === 'selector') {
      window.document.title = 'Choose an App - Hello World'
    }
  }, [currentView])

  return (
    <AppShell>
      {currentView === 'selector' ? (
        <AppSelector onLaunchTextEditor={() => setCurrentView('text-editor')} />
      ) : (
        <TextEditorApp
          document={textEditorDocument}
          onDocumentChange={setTextEditorDocument}
          onBackToApps={() => setCurrentView('selector')}
        />
      )}
    </AppShell>
  )
}

export default App
