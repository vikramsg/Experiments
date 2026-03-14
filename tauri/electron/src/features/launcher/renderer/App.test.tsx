import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { App } from './App'

describe('Launcher App', () => {
  it('shows browser, notes, and opencode launch cards', () => {
    render(
      <App
        openWorkspace={vi.fn().mockResolvedValue(undefined)}
        openOpenCode={vi.fn().mockResolvedValue(undefined)}
      />,
    )

    expect(screen.getByRole('heading', { name: /choose an app/i })).toBeVisible()
    expect(screen.getByRole('heading', { level: 2, name: /browser \+ notes/i })).toBeVisible()
    expect(screen.getByRole('button', { name: /launch browser \+ notes/i })).toBeVisible()
    expect(screen.getByRole('heading', { level: 2, name: /opencode/i })).toBeVisible()
    expect(screen.getByRole('button', { name: /launch opencode/i })).toBeVisible()
  })

  it('opens the workspace when the launch button is clicked', async () => {
    const user = userEvent.setup()
    const openWorkspace = vi.fn().mockResolvedValue(undefined)

    render(<App openWorkspace={openWorkspace} openOpenCode={vi.fn().mockResolvedValue(undefined)} />)
    await user.click(screen.getByRole('button', { name: /launch browser \+ notes/i }))

    expect(openWorkspace).toHaveBeenCalledTimes(1)
  })

  it('opens the OpenCode app when its launch button is clicked', async () => {
    const user = userEvent.setup()
    const openOpenCode = vi.fn().mockResolvedValue(undefined)

    render(<App openWorkspace={vi.fn().mockResolvedValue(undefined)} openOpenCode={openOpenCode} />)
    await user.click(screen.getByRole('button', { name: /launch opencode/i }))

    expect(openOpenCode).toHaveBeenCalledTimes(1)
  })
})
