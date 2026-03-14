import { fireEvent, render, screen } from '@testing-library/react'

import { SplitterHandle } from './SplitterHandle'

describe('SplitterHandle', () => {
  it('emits horizontal drag deltas while dragging', () => {
    const onAdjust = vi.fn()

    render(<SplitterHandle onAdjust={onAdjust} />)

    const separator = screen.getByRole('separator', { name: /resize panes/i })
    fireEvent.pointerDown(separator, { pointerId: 1, screenX: 200 })
    fireEvent.pointerMove(separator, { pointerId: 1, screenX: 224 })
    fireEvent.pointerMove(separator, { pointerId: 1, screenX: 236 })
    fireEvent.pointerUp(separator, { pointerId: 1, screenX: 236 })

    expect(onAdjust).toHaveBeenNthCalledWith(1, 24)
    expect(onAdjust).toHaveBeenNthCalledWith(2, 12)
  })

  it('stops emitting deltas after pointer up', () => {
    const onAdjust = vi.fn()

    render(<SplitterHandle onAdjust={onAdjust} />)

    const separator = screen.getByRole('separator', { name: /resize panes/i })
    fireEvent.pointerDown(separator, { pointerId: 1, screenX: 200 })
    fireEvent.pointerUp(separator, { pointerId: 1, screenX: 200 })
    fireEvent.pointerMove(separator, { pointerId: 1, screenX: 260 })

    expect(onAdjust).not.toHaveBeenCalled()
  })
})
