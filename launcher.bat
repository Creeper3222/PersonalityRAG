@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

echo [PersonalityRAG] Windows v0.1.0 launcher

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "RUNTIME_LOCK=requirements-runtime.lock"
set "DEPENDENCY_MARKER=.venv\.personalityrag-dependencies.json"

if exist "%VENV_PYTHON%" (
  "%VENV_PYTHON%" tools\runtime_bootstrap.py validate
  if errorlevel 1 (
    echo [PersonalityRAG] Existing .venv is not CPython 3.12 x64.
    echo [PersonalityRAG] The launcher will not delete or overwrite it. Preserve or rename it, then create a Python 3.12 environment.
    pause
    exit /b 1
  )
) else (
  echo [PersonalityRAG] Creating Python 3.12 virtual environment...
  py -3.12 -c "import struct,sys;raise SystemExit(0 if sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 else 1)"
  if errorlevel 1 (
    echo [PersonalityRAG] CPython 3.12 x64 was not found. Please install it first.
    pause
    exit /b 1
  )
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo [PersonalityRAG] Python 3.12 virtual environment creation failed.
    pause
    exit /b 1
  )
  "%VENV_PYTHON%" tools\runtime_bootstrap.py validate
  if errorlevel 1 (
    echo [PersonalityRAG] The new virtual environment failed Python 3.12 x64 validation.
    pause
    exit /b 1
  )
)

if not exist "%RUNTIME_LOCK%" (
  echo [PersonalityRAG] Missing %RUNTIME_LOCK%. Runtime dependencies cannot be verified.
  pause
  exit /b 1
)

"%VENV_PYTHON%" tools\runtime_bootstrap.py check --marker "%DEPENDENCY_MARKER%" --lock "%RUNTIME_LOCK%" --requirements requirements.txt
if errorlevel 2 (
  echo [PersonalityRAG] Dependency fingerprint validation failed.
  pause
  exit /b 1
)
if errorlevel 1 (
  echo [PersonalityRAG] Installing verified runtime dependencies...
  "%VENV_PYTHON%" -m pip install --disable-pip-version-check -r "%RUNTIME_LOCK%"
  if errorlevel 1 (
    echo [PersonalityRAG] Dependency installation failed.
    pause
    exit /b 1
  )
  "%VENV_PYTHON%" tools\runtime_bootstrap.py mark --marker "%DEPENDENCY_MARKER%" --lock "%RUNTIME_LOCK%" --requirements requirements.txt
  if errorlevel 1 (
    echo [PersonalityRAG] Dependency marker could not be written.
    pause
    exit /b 1
  )
) else (
  echo [PersonalityRAG] Verified dependencies are current.
)

echo [PersonalityRAG] Starting WebUI...
"%VENV_PYTHON%" run.py
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
