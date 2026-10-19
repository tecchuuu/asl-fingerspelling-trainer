// Light / dark theme. Default follows the OS; the nav toggle stores an explicit choice.
const KEY = "theme";

function stored() {
  try {
    return localStorage.getItem(KEY); // "light" | "dark" | null
  } catch {
    return null; // storage blocked (private window etc.): just follow the OS
  }
}

export const systemTheme = () =>
  window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";

export const currentTheme = () => stored() ?? systemTheme();

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // not remembered across visits, still applied now
  }
}

/** Apply a remembered choice on startup (otherwise CSS follows the OS). */
export function initTheme() {
  const t = stored();
  if (t) document.documentElement.dataset.theme = t;
}
