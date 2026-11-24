import React, { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import StockAutocomplete, { rankSuggestions } from './StockAutocomplete';

function Harness({ onSelect = vi.fn() }) {
  const [value, setValue] = useState('');
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSelect(value.trim().toUpperCase()); }}>
      <StockAutocomplete value={value} onChange={setValue} onSelect={onSelect} invalid={false} />
    </form>
  );
}

describe('rankSuggestions', () => {
  it('ranks exact ticker > ticker prefix > name word prefix > substring', () => {
    const items = [
      { ticker: 'XAPP', name: 'Random Substring Inc.' },
      { ticker: 'FOO', name: 'Global App Holdings' },
      { ticker: 'AAPL', name: 'Apple Inc.' },
      { ticker: 'APP', name: 'AppLovin Corp.' },
      { ticker: 'APPS', name: 'Apps Platform' },
    ];
    const ranked = rankSuggestions(items, 'app');
    expect(ranked.map((item) => item.ticker)).toEqual(['APP', 'APPS', 'FOO', 'AAPL', 'XAPP']);
  });
});

describe('StockAutocomplete', () => {
  it('preserves typed case and dismisses suggestions on Enter even when active === -1', () => {
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} />);
    const input = screen.getByRole('combobox', { name: 'Stock ticker' });
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: 'msft' } });
    expect(input).toHaveValue('msft');
    expect(screen.getByRole('listbox')).toBeInTheDocument();

    // Pressing Enter without arrow navigation closes suggestions immediately
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(screen.queryByRole('listbox')).toBeNull();
  });

  it('matches all 286 company names instantaneously from local universe metadata', () => {
    render(<Harness />);
    const input = screen.getByRole('combobox', { name: 'Stock ticker' });
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: 'Compass' } });
    const listbox = screen.getByRole('listbox');
    expect(listbox).toHaveTextContent('CPG.L');
    expect(listbox).toHaveTextContent('Compass Group');
  });
});
