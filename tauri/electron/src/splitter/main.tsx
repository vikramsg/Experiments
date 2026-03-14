import { createRoot } from 'react-dom/client'

import { SplitterHandle } from './SplitterHandle'

createRoot(document.getElementById('root')!).render(
  <SplitterHandle onAdjust={(delta) => void window.workspace.adjustSplitter(delta)} />,
)
