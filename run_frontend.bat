@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  run_frontend.bat  —  Launch the React + Tailwind Frontend (Port 5173)
REM ─────────────────────────────────────────────────────────────────────────────

cd /d "%~dp0frontend"

echo.
echo  ┌──────────────────────────────────────────────────────────┐
echo  │  AI Consumer Electronics Quality Dashboard               │
echo  │  http://localhost:5173 (React + Tailwind CSS UI)        │
echo  └──────────────────────────────────────────────────────────┘
echo.

npm run dev
