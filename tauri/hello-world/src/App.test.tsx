import { fireEvent, render, screen } from '@testing-library/react'

import App from './App'

describe('app shell', () => {
  it('starts on the app selector and hides editor actions', () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /choose an app/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /launch text editor/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /new/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /open/i })).not.toBeInTheDocument()
  })

  it('opens the text editor after choosing it and lets you return to apps', () => {
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /launch text editor/i }))

    expect(screen.getByRole('heading', { name: /text editor/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /back to apps/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /new/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /back to apps/i }))

    expect(screen.getByRole('heading', { name: /choose an app/i })).toBeInTheDocument()
  })
})
