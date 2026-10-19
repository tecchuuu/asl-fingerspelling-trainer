import { useEffect, useRef, useState } from "react";
import { getLandmarker, HAND_CONNECTIONS } from "../lib/landmarker.js";

/**
 * Webcam + MediaPipe + skeleton overlay, shared by every mode.
 * Calls onFrame({ landmarks, width, height, video }) for every new video frame;
 * landmarks is null when no hand is found. Detection runs on the raw (unmirrored) frame,
 * like the Python side; only the display is mirrored, like a mirror.
 */
export default function Camera({ onFrame }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const onFrameRef = useRef(onFrame);
  const [status, setStatus] = useState("Starting camera...");
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0); // bumped by the Retry button to restart the effect
  onFrameRef.current = onFrame;

  useEffect(() => {
    let stream;
    let rafId;
    let stopped = false;
    let lastTime = -1;

    async function start() {
      setFailed(false);
      setStatus("Starting camera...");
      try {
        stream = await openCamera(setStatus);
      } catch (err) {
        if (stopped) return;
        setStatus(err.name === "NotAllowedError"
          ? "Camera permission was denied. Allow it in the address bar (camera icon), then retry."
          : `Camera unavailable (${err.name}: ${err.message}). Close other apps or tabs using the camera, then retry.`);
        setFailed(true);
        return;
      }
      if (stopped) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      setStatus("Loading hand model...");
      const landmarker = await getLandmarker();
      if (stopped) return;
      setStatus("");

      const loop = () => {
        if (stopped) return;
        if (video.currentTime !== lastTime) {
          lastTime = video.currentTime;
          const result = landmarker.detectForVideo(video, performance.now());
          const landmarks = result.landmarks?.[0] ?? null;
          draw(canvasRef.current, video, landmarks);
          onFrameRef.current?.({ landmarks, width: video.videoWidth, height: video.videoHeight, video });
        }
        rafId = requestAnimationFrame(loop);
      };
      loop();
    }

    start();
    return () => {
      stopped = true;
      cancelAnimationFrame(rafId);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [attempt]);

  return (
    <div className="camera">
      <video ref={videoRef} playsInline muted />
      <canvas ref={canvasRef} />
      {status && (
        <div className="camera-status">
          <div>
            <p>{status}</p>
            {failed && <button onClick={() => setAttempt((a) => a + 1)}>Retry camera</button>}
          </div>
        </div>
      )}
    </div>
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Right after the permission prompt, Chrome's own camera preview can still hold the webcam,
// and some Windows webcams refuse a second open ("Could not start video source", NotReadableError).
// So: preferred settings first, then defaults, waiting a little longer before each retry.
const ATTEMPTS = [
  { delay: 0, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } },
  { delay: 700, video: true },
  { delay: 1500, video: true },
  { delay: 3000, video: true },
];

async function openCamera(onStatus) {
  let lastError;
  for (const [i, { delay, video }] of ATTEMPTS.entries()) {
    if (delay) {
      onStatus(`Camera busy, retrying (${i}/${ATTEMPTS.length - 1})...`);
      await sleep(delay);
    }
    try {
      return await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (err) {
      if (err.name === "NotAllowedError") throw err; // denied: retrying won't help
      console.warn(`Camera attempt ${i + 1} failed:`, err);
      lastError = err;
    }
  }
  throw lastError;
}

function draw(canvas, video, landmarks) {
  const w = (canvas.width = video.videoWidth);
  const h = (canvas.height = video.videoHeight);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (!landmarks) return;
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 3;
  for (const { start, end } of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[start].x * w, landmarks[start].y * h);
    ctx.lineTo(landmarks[end].x * w, landmarks[end].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ef4444";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}
