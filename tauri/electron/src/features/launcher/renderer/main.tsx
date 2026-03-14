import { createRoot } from 'react-dom/client'

import { App } from './App'

createRoot(document.getElementById('root')!).render(
  <App openWorkspace={() => window.launcher.openWorkspace()} openOpenCode={() => window.launcher.openOpenCode()} />,
)
