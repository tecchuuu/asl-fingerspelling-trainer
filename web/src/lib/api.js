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
