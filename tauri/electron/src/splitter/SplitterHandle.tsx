import type { CSSProperties } from 'react'

export type SplitterHandleProps = {
  onAdjust: (delta: number) => void
}

export function SplitterHandle({ onAdjust }: SplitterHandleProps) {
  let lastScreenX = 0
  let dragging = false

  return (
    <div
      role="separator"
      aria-label="Resize panes"
      aria-orientation="vertical"
      style={styles.root}
      onPointerDown={(event) => {
        dragging = true
        lastScreenX = event.screenX
        event.currentTarget.setPointerCapture?.(event.pointerId)
      }}
      onPointerMove={(event) => {
        if (!dragging) {
          return
        }

        const delta = event.screenX - lastScreenX
        lastScreenX = event.screenX

        if (delta !== 0) {
          onAdjust(delta)
        }
      }}
      onPointerUp={(event) => {
        dragging = false
        event.currentTarget.releasePointerCapture?.(event.pointerId)
      }}
    >
      <div style={styles.handle} />
    </div>
  )
}

const styles: Record<string, CSSProperties> = {
  root: {
    width: '100%',
    height: '100vh',
    background: 'linear-gradient(180deg, #d7d3c7 0%, #c9c4b4 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'col-resize',
    userSelect: 'none',
  },
  handle: {
    width: '4px',
    height: '72px',
    borderRadius: '999px',
    background: '#645e50',
  },
}
