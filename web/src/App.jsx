import { useState } from "react";
import CheckPage from "./pages/CheckPage.jsx";
import SpellPage from "./pages/SpellPage.jsx";

const PAGES = {
  spell: { label: "Spell", component: SpellPage },
  check: { label: "Browser check", component: CheckPage, dev: true },
};

export default function App() {
  const [page, setPage] = useState("spell");
  const Page = PAGES[page].component;
  return (
    <>
      <nav className="nav">
        <span className="brand">✋ Fingerspelling Trainer</span>
        {Object.entries(PAGES).map(([key, p]) => (
          <button key={key} className={`${page === key ? "on" : ""} ${p.dev ? "dev" : ""}`} onClick={() => setPage(key)}>
            {p.label}
          </button>
        ))}
      </nav>
      {/* key: switching pages unmounts the old one, which releases the camera */}
      <Page key={page} />
    </>
  );
}
