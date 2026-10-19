// Calls to the FastAPI server. Landmarks are sent as MediaPipe gives them: [[x, y, z] x 21].

const toPoints = (landmarks) => landmarks.map((p) => [p.x, p.y, p.z]);

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

export const predict = (landmarks, width, height) =>
  post("/api/predict", { landmarks: toPoints(landmarks), width, height });

export const compareWithPython = (landmarks, width, height, image) =>
  post("/api/debug/compare", { landmarks: toPoints(landmarks), width, height, image });

// --- Attempt log ---

export const startSession = (mode, category, itemsTotal = null) =>
  post("/api/sessions", { mode, category, items_total: itemsTotal }).then((r) => r.session_id);

export const finishSession = (sessionId) => post(`/api/sessions/${sessionId}/finish`, {});

/** Fire-and-forget: a failed log write must never interrupt practice. */
export function logAttempt(attempt) {
  post("/api/attempts", attempt).catch((err) => console.warn("attempt not logged:", err));
}

export async function getProgress() {
  const res = await fetch("/api/progress");
  if (!res.ok) throw new Error(`/api/progress -> HTTP ${res.status}`);
  return res.json();
}
