import { useState } from "react";
import CheckPage from "./pages/CheckPage.jsx";
import ExamPage from "./pages/ExamPage.jsx";
import HomePage from "./pages/HomePage.jsx";
import PracticePage from "./pages/PracticePage.jsx";

// view: { page: "home" } | { page: "practice" | "exam", categoryId } | { page: "check" }
export default function App() {
  const [view, setView] = useState({ page: "home" });
  const home = () => setView({ page: "home" });

  return (
    <>
      <nav className="nav">
        <button className="brand" onClick={home}>✋ Fingerspelling Trainer</button>
        <button className={["home", "practice", "exam"].includes(view.page) ? "on" : ""} onClick={home}>
          Categories
        </button>
        <button className={`dev ${view.page === "check" ? "on" : ""}`} onClick={() => setView({ page: "check" })}>
          Browser check
        </button>
      </nav>
      {/* key: changing view unmounts the old page, which releases the camera */}
      {view.page === "home" && <HomePage onStart={(categoryId, mode) => setView({ page: mode, categoryId })} />}
      {view.page === "practice" && <PracticePage key={view.categoryId} categoryId={view.categoryId} onBack={home} />}
      {view.page === "exam" && (
        <ExamPage key={view.categoryId} categoryId={view.categoryId} onBack={home}
          onPractice={() => setView({ page: "practice", categoryId: view.categoryId })} />
      )}
      {view.page === "check" && <CheckPage />}
    </>
  );
}
