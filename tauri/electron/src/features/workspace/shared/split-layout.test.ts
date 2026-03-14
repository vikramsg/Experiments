import { clampNotesWidth, computeSplitLayout, type SplitLayoutInput } from './split-layout'

describe('split-layout', () => {
  const baseInput: SplitLayoutInput = {
    windowWidth: 1280,
    windowHeight: 800,
    notesWidth: 420,
    splitterWidth: 12,
    minNotesWidth: 280,
    minBrowserWidth: 360,
  }

  it('keeps an in-range notes width', () => {
    expect(clampNotesWidth(baseInput)).toBe(420)
  })

  it('clamps the notes pane to the minimum width', () => {
    expect(clampNotesWidth({ ...baseInput, notesWidth: 120 })).toBe(280)
  })

  it('clamps the notes pane so the browser keeps its minimum width', () => {
    expect(clampNotesWidth({ ...baseInput, notesWidth: 1200 })).toBe(908)
  })

  it('computes the browser width from the remaining space', () => {
    expect(computeSplitLayout(baseInput)).toEqual({
      notesWidth: 420,
      browserWidth: 848,
      splitterX: 420,
    })
  })
})
