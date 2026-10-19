// MediaPipe needs its WebAssembly files and the hand model served as plain files.
// Copy them into public/ so the app works offline (no CDN) and uses the same model as Python.
import { cpSync, mkdirSync } from "node:fs";

mkdirSync("public/mediapipe", { recursive: true });
cpSync("node_modules/@mediapipe/tasks-vision/wasm", "public/mediapipe/wasm", { recursive: true });
cpSync("../models/hand_landmarker.task", "public/mediapipe/hand_landmarker.task");
console.log("Copied MediaPipe wasm + hand_landmarker.task into public/mediapipe");
