import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

// Node 26 exposes its own storage globals; browser tests must use jsdom's
// origin-scoped storage, not the server runtime's file-backed implementation.
vi.stubGlobal('localStorage', jsdom.window.localStorage);
vi.stubGlobal('sessionStorage', jsdom.window.sessionStorage);

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

if (!window.matchMedia) {
  window.matchMedia = () => ({
    matches: false,
    media: '',
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}
