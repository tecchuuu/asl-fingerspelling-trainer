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
  onFrameRef.current = onFrame;

  useEffect(() => {
    let stream;
    let rafId;
    let stopped = false;
    let lastTime = -1;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30 } },
          audio: false,
        });
      } catch (err) {
        setStatus(`Camera unavailable: ${err.message}`);
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
  }, []);

  return (
    <div className="camera">
      <video ref={videoRef} playsInline muted />
      <canvas ref={canvasRef} />
      {status && <div className="camera-status">{status}</div>}
    </div>
  );
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
