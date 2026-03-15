import { FitAddon, Ghostty, Terminal } from 'ghostty-web'
import ghosttyWasmUrl from 'ghostty-web/ghostty-vt.wasm?url'

import { toGhosttyWebFontFamily, type TerminalAppearance } from '../../../terminal-model'

type GhosttyRuntimeInput = {
  container: HTMLElement
  cols: number
  rows: number
  appearance: TerminalAppearance
  onInput: (data: string) => void
  onResize: (cols: number, rows: number) => void
}

export type GhosttyRuntime = {
  write: (data: string) => void
  focus: () => void
  dispose: () => void
  getSize: () => { cols: number; rows: number }
}

let ghosttyPromise: Promise<Ghostty> | null = null

function loadGhostty() {
  ghosttyPromise ??= Ghostty.load(ghosttyWasmUrl)
  return ghosttyPromise
}

export async function createGhosttyRuntime(input: GhosttyRuntimeInput): Promise<GhosttyRuntime> {
  const ghostty = await loadGhostty()
  const terminal = new Terminal({
    cols: input.cols,
    rows: input.rows,
    ghostty,
    cursorBlink: true,
    fontSize: input.appearance.fontSize,
    fontFamily: toGhosttyWebFontFamily(input.appearance.fontFamily),
    theme: {
      background: '#101519',
      foreground: '#d8e5ea',
      cursor: '#f4d38d',
      cursorAccent: '#101519',
      selectionBackground: '#29414a',
      black: '#0f1418',
      red: '#ef8b80',
      green: '#8fcf9a',
      yellow: '#eacb7a',
      blue: '#7dbad8',
      magenta: '#c39be5',
      cyan: '#79d1cb',
      white: '#d8e5ea',
      brightBlack: '#5e7077',
      brightRed: '#f7a39a',
      brightGreen: '#a9dfb1',
      brightYellow: '#f0d899',
      brightBlue: '#99cae3',
      brightMagenta: '#d3b4ef',
      brightCyan: '#95dfda',
      brightWhite: '#edf5f7',
    },
  })

  const fitAddon = new FitAddon()
  const inputDisposable = terminal.onData((data) => {
    input.onInput(data)
  })
  const resizeDisposable = terminal.onResize(({ cols, rows }) => {
    input.onResize(cols, rows)
  })

  terminal.loadAddon(fitAddon)
  terminal.open(input.container)
  fitAddon.fit()
  fitAddon.observeResize()

  return {
    write: (data: string) => {
      terminal.write(data)
    },
    focus: () => {
      terminal.focus()
    },
    dispose: () => {
      inputDisposable.dispose()
      resizeDisposable.dispose()
      fitAddon.dispose()
      terminal.dispose()
    },
    getSize: () => ({
      cols: terminal.cols,
      rows: terminal.rows,
    }),
  }
}
