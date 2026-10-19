// MediaPipe hand landmarker in the browser: same model file and version (1.1.1) as the Python side.
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

let landmarkerPromise = null;

export function getLandmarker() {
  // Created once and shared by every page; loading the model takes a second or two
  landmarkerPromise ??= FilesetResolver.forVisionTasks("/mediapipe/wasm").then((vision) =>
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "/mediapipe/hand_landmarker.task" },
      runningMode: "VIDEO",
      numHands: 1,
    }),
  );
  return landmarkerPromise;
}

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
