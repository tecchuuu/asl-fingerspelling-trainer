@echo off
rem Starts the API and the React dev server in two windows. Close a window to stop that server.
rem Then open http://127.0.0.1:5173 (127.0.0.1, not localhost: localhost adds ~2 s per request here).
cd /d "%~dp0"
start "API (FastAPI)" cmd /k ".venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
start "Web (Vite)" cmd /k "cd web && npm run dev"
