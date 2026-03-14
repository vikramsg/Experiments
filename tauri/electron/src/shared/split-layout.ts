export type SplitLayoutInput = {
  windowWidth: number
  windowHeight: number
  notesWidth: number
  splitterWidth: number
  minNotesWidth: number
  minBrowserWidth: number
}

export type SplitLayout = {
  notesWidth: number
  browserWidth: number
  splitterX: number
}

export function clampNotesWidth(input: SplitLayoutInput): number {
  const maxNotesWidth = Math.max(
    input.minNotesWidth,
    input.windowWidth - input.splitterWidth - input.minBrowserWidth,
  )

  return Math.min(Math.max(input.notesWidth, input.minNotesWidth), maxNotesWidth)
}

export function computeSplitLayout(input: SplitLayoutInput): SplitLayout {
  const notesWidth = clampNotesWidth(input)

  return {
    notesWidth,
    browserWidth: input.windowWidth - notesWidth - input.splitterWidth,
    splitterX: notesWidth,
  }
}
