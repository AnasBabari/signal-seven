import { describe, expect, it } from 'vitest';
import { JSDOM } from 'jsdom';

describe('jsdom 30 runtime compatibility', () => {
  it('uses browser storage globals rather than Node storage', () => {
    expect(globalThis.localStorage).toBe(jsdom.window.localStorage);
    expect(globalThis.sessionStorage).toBe(jsdom.window.sessionStorage);
    localStorage.setItem('runtime-check', 'ok');
    expect(window.localStorage.getItem('runtime-check')).toBe('ok');
    localStorage.removeItem('runtime-check');
  });

  it('supports selectors, computed styles, focus and click interactions', () => {
    const dom = new JSDOM(`<!doctype html>
      <style>button[data-action="save"] { color: rgb(1, 2, 3); }</style>
      <button data-action="save">Save</button>`, {
      url: 'https://example.test/',
    });
    try {
      const { window } = dom;
      const button = window.document.querySelector('button[data-action="save"]');
      expect(window.getComputedStyle(button).color).toBe('rgb(1, 2, 3)');
      button.focus();
      expect(window.document.activeElement).toBe(button);
      let clicks = 0;
      button.addEventListener('click', () => { clicks += 1; });
      button.click();
      expect(clicks).toBe(1);
      expect(new window.URL('/outlook', window.location.href).pathname).toBe('/outlook');
    } finally {
      dom.window.close();
    }
  });
});
