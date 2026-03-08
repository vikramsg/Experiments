import CodeMirror from '@uiw/react-codemirror'
import { markdown } from '@codemirror/lang-markdown'

type CodeEditorProps = {
  value: string
  onChange: (value: string) => void
}

const extensions = [markdown()]

export function CodeEditor({ value, onChange }: CodeEditorProps) {
  return (
    <div className="editor-canvas" data-testid="editor-canvas">
      <CodeMirror
        value={value}
        height="100%"
        theme="light"
        extensions={extensions}
        basicSetup={{
          foldGutter: false,
          lineNumbers: false,
          highlightActiveLineGutter: false,
        }}
        onChange={onChange}
      />
    </div>
  )
}
