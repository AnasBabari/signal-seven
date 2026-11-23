import React, { useEffect, useState } from 'react';
import { ALL_VALID_TICKERS, ALL_TICKERS_SET } from '../universe';
import { requestJson } from '../api/request';

const LOCAL_ITEMS = ALL_VALID_TICKERS.map((ticker) => ({ ticker, name: ticker }));

export function rankSuggestions(items, rawQuery) {
  const query = String(rawQuery || '').trim().toUpperCase();
  if (!query) return [];
  const seen = new Set();
  const scored = [];
  for (const item of items) {
    const ticker = String(item?.ticker || '').trim().toUpperCase();
    if (!ticker || seen.has(ticker)) continue;
    seen.add(ticker);
    const name = String(item?.name || ticker).trim();
    const upperName = name.toUpperCase();
    const words = upperName.split(/[^A-Z0-9]+/).filter(Boolean);
    let rank = 99;
    if (ticker === query) rank = 0;
    else if (ticker.startsWith(query)) rank = 1;
    else if (words.some((word) => word.startsWith(query))) rank = 2;
    else if (ticker.includes(query) || upperName.includes(query)) rank = 3;
    else continue;
    scored.push({ ticker, name, rank });
  }
  scored.sort(
    (a, b) => a.rank - b.rank || a.ticker.length - b.ticker.length || a.ticker.localeCompare(b.ticker),
  );
  return scored.slice(0, 6).map(({ ticker, name }) => ({ ticker, name }));
}

export default function StockAutocomplete({ value, onChange, onSelect, invalid, errorId }) {
  const [suggestions, setSuggestions] = useState([]);
  const [active, setActive] = useState(-1);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setActive(-1);
    const query = value.trim();
    if (!query || !open) {
      if (!query) setSuggestions([]);
      return undefined;
    }
    const localRanked = rankSuggestions(LOCAL_ITEMS, query);
    setSuggestions(localRanked);
    const controller = new AbortController();
    const timer = setTimeout(() => {
      requestJson(`/api/v1/search?query=${encodeURIComponent(query)}`, { signal: controller.signal, timeoutMs: 5_000 })
        .then((result) => {
          if (controller.signal.aborted) return;
          const remoteSupported = (result.results || []).filter((item) => ALL_TICKERS_SET.has(String(item?.ticker || '').toUpperCase()));
          setSuggestions(rankSuggestions([...remoteSupported, ...localRanked], query));
        })
        .catch(() => {});
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [value, open]);

  const choose = (ticker) => {
    const normalized = String(ticker || '').trim().toUpperCase();
    onChange(normalized);
    setOpen(false);
    setActive(-1);
    onSelect(normalized);
  };

  return (
    <>
      <input
        id="tickerInput"
        role="combobox"
        aria-label="Stock ticker"
        value={value}
        onChange={(event) => { onChange(event.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        placeholder="Enter a ticker or company name"
        maxLength={100}
        autoComplete="off"
        spellCheck="false"
        aria-invalid={invalid}
        aria-describedby={errorId}
        aria-autocomplete="list"
        aria-controls="stock-suggestions"
        aria-expanded={open && suggestions.length > 0}
        aria-activedescendant={active >= 0 && open ? `stock-option-${active}` : undefined}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            setOpen(false);
            setActive(-1);
          } else if (['ArrowDown', 'ArrowUp'].includes(event.key) && suggestions.length) {
            event.preventDefault();
            setOpen(true);
            setActive((index) => (index + (event.key === 'ArrowDown' ? 1 : suggestions.length - 1) + suggestions.length) % suggestions.length);
          } else if (event.key === 'Enter') {
            if (active >= 0 && open && suggestions[active]) {
              event.preventDefault();
              choose(suggestions[active].ticker);
            } else {
              setOpen(false);
              setActive(-1);
            }
          }
        }}
      />
      {open && suggestions.length > 0 && (
        <ul id="stock-suggestions" role="listbox" aria-label="Stock suggestions" className="stock-suggestions">
          {suggestions.map((item, index) => (
            <li
              key={item.ticker}
              id={`stock-option-${index}`}
              role="option"
              aria-selected={index === active}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(item.ticker)}
            >
              <strong>{item.ticker}</strong>
              {item.name !== item.ticker && <span>{item.name}</span>}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
