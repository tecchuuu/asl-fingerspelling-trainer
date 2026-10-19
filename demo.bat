@echo off
rem Demo: builds the React app, then serves everything from one server.
rem Open http://127.0.0.1:8000 once it says "Uvicorn running".
cd /d "%~dp0"
cd web
call npm run build || (echo Build failed & pause & exit /b 1)
cd ..
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
