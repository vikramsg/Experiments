import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { BrowserApi } from '../../../browser-contract'
import type { BrowserSnapshot } from '../../../browser-model'
import { App } from './App'

function createApi(snapshot: BrowserSnapshot): BrowserApi {
  const api: BrowserApi = {
    loadState: vi.fn().mockResolvedValue(snapshot),
    setBrowserUrl: vi.fn().mockResolvedValue(undefined),
    goBack: vi.fn().mockResolvedValue(undefined),
    goForward: vi.fn().mockResolvedValue(undefined),
    onStateChange: vi.fn().mockReturnValue(() => undefined),
  }

  return api
}

describe('Browser Chrome App', () => {
  it('loads the saved browser url and navigates on go', async () => {
    const user = userEvent.setup()
    const api = createApi({
      browserUrl: 'https://example.com',
      canGoBack: false,
      canGoForward: false,
    })

    render(<App api={api} />)

    const input = await screen.findByRole('textbox', { name: /browser url/i })
    expect(input).toHaveValue('https://example.com')
    expect(screen.getByRole('button', { name: /back/i })).toHaveTextContent('←')
    expect(screen.getByRole('button', { name: /forward/i })).toHaveTextContent('→')
    expect(screen.queryByText(/^Back$/)).not.toBeInTheDocument()
    expect(screen.queryByText(/^Forward$/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /back/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /forward/i })).toBeDisabled()

    await user.clear(input)
    await user.type(input, 'https://example.org/docs')
    await user.click(screen.getByRole('button', { name: /^go$/i }))

    expect(api.setBrowserUrl).toHaveBeenCalledWith('https://example.org/docs')
  })

  it('calls the back and forward browser actions', async () => {
    const user = userEvent.setup()
    const api = createApi({
      browserUrl: 'https://example.com',
      canGoBack: true,
      canGoForward: true,
    })

    render(<App api={api} />)

    await user.click(await screen.findByRole('button', { name: /back/i }))
    await user.click(screen.getByRole('button', { name: /forward/i }))

    expect(api.goBack).toHaveBeenCalledTimes(1)
    expect(api.goForward).toHaveBeenCalledTimes(1)
  })

  it('reacts to workspace state updates', async () => {
    const api = createApi({
      browserUrl: 'https://example.com',
      canGoBack: false,
      canGoForward: false,
    })

    let listener: ((snapshot: BrowserSnapshot) => void) | undefined
    vi.mocked(api.onStateChange).mockImplementation((nextListener) => {
      listener = nextListener
      return () => undefined
    })

    render(<App api={api} />)

    const input = await screen.findByRole('textbox', { name: /browser url/i })

    act(() => {
      listener?.({
        browserUrl: 'https://example.net/app',
        canGoBack: true,
        canGoForward: false,
      })
    })

    expect(input).toHaveValue('https://example.net/app')
    expect(screen.getByRole('button', { name: /back/i })).toBeEnabled()
    expect(screen.getByRole('button', { name: /forward/i })).toBeDisabled()
  })
})
