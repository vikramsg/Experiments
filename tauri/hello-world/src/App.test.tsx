import { render, screen } from '@testing-library/react'

import App from './App'

describe('app shell', () => {
  it('renders the text editor app view as the primary experience', () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /text editor/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /new/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /open/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^save$/i })).toBeInTheDocument()
    expect(screen.getAllByText(/untitled/i).length).toBeGreaterThan(0)
  })
})
