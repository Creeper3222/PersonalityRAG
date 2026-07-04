@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo [PersonalityRAG] Windows v0.1.0 launcher

if not exist ".venv\Scripts\python.exe" (
  echo [PersonalityRAG] Creating Python 3.12 virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo [PersonalityRAG] Python 3.12 x64 was not found. Please install it first.
    pause
    exit /b 1
  )
)

echo [PersonalityRAG] Checking dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo [PersonalityRAG] Dependency installation failed.
  pause
  exit /b 1
)

echo [PersonalityRAG] Starting WebUI...
".venv\Scripts\python.exe" run.py
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
