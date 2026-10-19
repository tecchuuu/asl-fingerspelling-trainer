import { useState } from "react";
import { applyTheme, currentTheme } from "./lib/theme.js";
import CheckPage from "./pages/CheckPage.jsx";
import ExamPage from "./pages/ExamPage.jsx";
import HomePage from "./pages/HomePage.jsx";
import PracticePage from "./pages/PracticePage.jsx";
import ProgressPage from "./pages/ProgressPage.jsx";
import TrainerPage from "./pages/TrainerPage.jsx";

// view: { page: "home" | "progress" | "trainer" | "check" } | { page: "practice" | "exam", categoryId }
export default function App() {
  const [view, setView] = useState({ page: "home" });
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
        <button className="brand" onClick={home}>✋ Fingerspelling Trainer</button>
        <button className={["home", "practice", "exam"].includes(view.page) ? "on" : ""} onClick={home}>
          Categories
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
      {view.page === "home" && <HomePage onStart={(categoryId, mode) => setView({ page: mode, categoryId })} />}
      {view.page === "practice" && <PracticePage key={view.categoryId} categoryId={view.categoryId} onBack={home} />}
      {view.page === "exam" && (
        <ExamPage key={view.categoryId} categoryId={view.categoryId} onBack={home}
          onPractice={() => setView({ page: "practice", categoryId: view.categoryId })} />
      )}
      {view.page === "progress" && <ProgressPage onStart={(categoryId, mode) => setView({ page: mode, categoryId })} />}
      {view.page === "trainer" && <TrainerPage />}
      {view.page === "check" && <CheckPage />}
    </>
  );
}
