@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  run_api.bat  —  Launch the FastAPI backend
REM  Uses the SYSTEM Python (where torch, fastapi, etc. are installed).
REM  If you want to use a venv instead, activate it BEFORE running this script.
REM ─────────────────────────────────────────────────────────────────────────────

REM Make sure we're in the project root regardless of where the script is called from
cd /d "%~dp0"

REM Add the project root to PYTHONPATH so 'src.*' imports resolve
set PYTHONPATH=%~dp0;%PYTHONPATH%

REM Default port (override with: set API_PORT=9000 && run_api.bat)
if not defined API_PORT set API_PORT=8000
if not defined API_HOST set API_HOST=0.0.0.0

echo.
echo  ┌──────────────────────────────────────────────────────────┐
echo  │  AI Consumer Electronics Quality Intelligence API         │
echo  │  http://localhost:%API_PORT%/docs  (Swagger UI)              │
echo  │  http://localhost:%API_PORT%/redoc (ReDoc)                   │
echo  └──────────────────────────────────────────────────────────┘
echo.

REM Use System Python where torch and transformers are installed
if exist "C:\Program Files\Python313\python.exe" (
    "C:\Program Files\Python313\python.exe" -m uvicorn src.api.main:app --host %API_HOST% --port %API_PORT% --reload
) else (
    python -m uvicorn src.api.main:app --host %API_HOST% --port %API_PORT% --reload
)
