// The current page lives in the URL hash (#/practice/food), so F5, Back/Forward and links work.
// A hash (not /practice/food paths) needs no server changes: FastAPI and Vite only ever serve index.html.
import { useCallback, useEffect, useState } from "react";
import { categoryById, LETTERS } from "./words.js";

const SIMPLE = ["learn", "free", "progress", "trainer", "check"];

export function viewToHash(view) {
  if (view.page === "practice" && view.letter) return `#/letter/${view.letter}`;
  if (view.page === "practice" || view.page === "exam") return `#/${view.page}/${view.categoryId}`;
  if (SIMPLE.includes(view.page)) return `#/${view.page}`;
  return "#/";
}

/** Parse the hash; anything unknown (old link, typo) falls back to the home page. */
export function hashToView(hash) {
  const [page, arg] = hash.replace(/^#\/?/, "").split("/");
  if (page === "letter" && LETTERS.includes(arg)) return { page: "practice", letter: arg };
  if ((page === "practice" || page === "exam") && categoryById(arg)) return { page, categoryId: arg };
  if (SIMPLE.includes(page)) return { page };
  return { page: "home" };
}

export function useRoute() {
  const [view, setViewState] = useState(() => hashToView(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setViewState(hashToView(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Setting the hash adds a history entry; the hashchange listener then updates the view
  const setView = useCallback((next) => {
    const hash = viewToHash(next);
    if (hash === window.location.hash || (hash === "#/" && !window.location.hash)) setViewState(next);
    else window.location.hash = hash;
  }, []);

  return [view, setView];
}
