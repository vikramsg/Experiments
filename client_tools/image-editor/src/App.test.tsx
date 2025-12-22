import { render, screen } from '@testing-library/react';
import App from './App';

describe('App', () => {
  it('renders the header and empty state', () => {
    render(<App />);

    expect(screen.getByText('Pro Image Compositor')).toBeInTheDocument();
    expect(screen.getByText('Upload images to begin')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /add image/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /export/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /pdf tools/i })).toBeInTheDocument();
  });
});
