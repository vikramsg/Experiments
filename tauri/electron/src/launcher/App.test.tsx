import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { App } from './App'

describe('Launcher App', () => {
  it('shows a browser and notes launch card', () => {
    render(<App openWorkspace={vi.fn()} />)

    expect(screen.getByRole('heading', { name: /choose an app/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /launch browser \+ notes/i })).toBeInTheDocument()
  })

  it('opens the workspace when the launch button is clicked', async () => {
    const user = userEvent.setup()
    const openWorkspace = vi.fn().mockResolvedValue(undefined)

    render(<App openWorkspace={openWorkspace} />)
    await user.click(screen.getByRole('button', { name: /launch browser \+ notes/i }))

    expect(openWorkspace).toHaveBeenCalledTimes(1)
  })
})
