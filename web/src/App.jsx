import { useState } from "react";
import HandSkeleton from "./components/HandSkeleton.jsx";
import { useRoute } from "./lib/route.js";
import { applyTheme, currentTheme } from "./lib/theme.js";
import CheckPage from "./pages/CheckPage.jsx";
import ExamPage from "./pages/ExamPage.jsx";
import FreePage from "./pages/FreePage.jsx";
import HomePage from "./pages/HomePage.jsx";
import LearnPage from "./pages/LearnPage.jsx";
import PracticePage from "./pages/PracticePage.jsx";
import ProgressPage from "./pages/ProgressPage.jsx";
import TrainerPage from "./pages/TrainerPage.jsx";

// view: { page: "home" | "learn" | "free" | "progress" | "trainer" | "check" } | { page: "practice" | "exam", categoryId }
//       | { page: "practice", letter } (one letter, from Learn)
export default function App() {
  const [view, setView] = useRoute(); // kept in the URL hash, see lib/route.js
  const home = () => setView({ page: "home" });
  const [theme, setTheme] = useState(currentTheme);
  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    setTheme(next);
  };

  return (
    <>
      <nav className="nav">
        <button className="brand" onClick={home} aria-label="Fingerspell, home">
          <HandSkeleton letter="B" size={30} />
          <span>Fingerspell</span>
        </button>
        <button className={["home", "practice", "exam"].includes(view.page) && !view.letter ? "on" : ""} onClick={home}>
          Home
        </button>
        <button className={view.page === "learn" || view.letter ? "on" : ""} onClick={() => setView({ page: "learn" })}>
          Learn
        </button>
        <button className={view.page === "free" ? "on" : ""} onClick={() => setView({ page: "free" })}>
          Free spell
        </button>
        <button className={view.page === "progress" ? "on" : ""} onClick={() => setView({ page: "progress" })}>
          My progress
        </button>
        <button className={view.page === "trainer" ? "on" : ""} onClick={() => setView({ page: "trainer" })}>
          Trainer
        </button>
        <button className={`dev ${view.page === "check" ? "on" : ""}`} onClick={() => setView({ page: "check" })}>
          Browser check
        </button>
        <button className="theme-toggle" onClick={toggleTheme}
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
      </nav>
      {/* key: changing view unmounts the old page, which releases the camera */}
      {view.page === "home" && (
        <HomePage onStart={(categoryId, mode) => setView({ page: mode, categoryId })} onLearn={() => setView({ page: "learn" })} />
      )}
      {view.page === "learn" && <LearnPage onTry={(letter) => setView({ page: "practice", letter })} />}
      {view.page === "practice" && (
        <PracticePage key={view.letter ?? view.categoryId} categoryId={view.categoryId} letter={view.letter}
          onBack={view.letter ? () => setView({ page: "learn" }) : home} />
      )}
      {view.page === "exam" && (
        <ExamPage key={view.categoryId} categoryId={view.categoryId} onBack={home}
          onPractice={() => setView({ page: "practice", categoryId: view.categoryId })} />
      )}
      {view.page === "free" && <FreePage />}
      {view.page === "progress" && <ProgressPage onStart={(categoryId, mode) => setView({ page: mode, categoryId })} />}
      {view.page === "trainer" && <TrainerPage />}
      {view.page === "check" && <CheckPage />}
    </>
  );
}
