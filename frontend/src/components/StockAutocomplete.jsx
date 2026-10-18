import React, { useEffect, useState } from 'react';
import { ALL_VALID_TICKERS, ALL_TICKERS_SET } from '../universe';
import { requestJson } from '../api/request';

export default function StockAutocomplete({ value, onChange, onSelect, invalid, errorId }) {
  const [suggestions, setSuggestions] = useState([]);
  const [active, setActive] = useState(-1);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    setActive(-1);
    const query = value.trim();
    if (!query) { setSuggestions([]); return undefined; }
    setSuggestions(ALL_VALID_TICKERS.filter((ticker) => ticker.includes(query.toUpperCase())).slice(0, 6).map((ticker) => ({ ticker, name: ticker })));
    const controller = new AbortController();
    const timer = setTimeout(() => {
      requestJson(`/api/v1/search?query=${encodeURIComponent(query)}`, { signal: controller.signal, timeoutMs: 5_000 })
        .then((result) => { if (!controller.signal.aborted) setSuggestions((result.results || []).filter((item) => ALL_TICKERS_SET.has(item.ticker)).slice(0, 6)); })
        .catch(() => {});
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [value]);
  const choose = (ticker) => { onChange(ticker); setOpen(false); onSelect(ticker); };
  return <>
    <input id="tickerInput" role="combobox" aria-label="Stock ticker" value={value} onChange={(event) => { onChange(event.target.value.toUpperCase()); setOpen(true); }}
      onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} placeholder="Enter a ticker or company name" maxLength={100} autoComplete="off" spellCheck="false"
      aria-invalid={invalid} aria-describedby={errorId} aria-autocomplete="list" aria-controls="stock-suggestions" aria-expanded={open && suggestions.length > 0}
      aria-activedescendant={active >= 0 && open ? `stock-option-${active}` : undefined}
      onKeyDown={(event) => {
        if (event.key === 'Escape') { setOpen(false); setActive(-1); }
        if (['ArrowDown', 'ArrowUp'].includes(event.key) && suggestions.length) { event.preventDefault(); setOpen(true); setActive((index) => (index + (event.key === 'ArrowDown' ? 1 : suggestions.length - 1) + suggestions.length) % suggestions.length); }
        if (event.key === 'Enter' && active >= 0 && open) { event.preventDefault(); choose(suggestions[active].ticker); }
      }} />
    {open && suggestions.length > 0 && <ul id="stock-suggestions" role="listbox" aria-label="Stock suggestions" className="stock-suggestions">
      {suggestions.map((item, index) => <li key={item.ticker} id={`stock-option-${index}`} role="option" aria-selected={index === active}
        onMouseDown={(event) => event.preventDefault()} onClick={() => choose(item.ticker)}><strong>{item.ticker}</strong> {item.name !== item.ticker && <span>{item.name}</span>}</li>)}
    </ul>}
  </>;
}
