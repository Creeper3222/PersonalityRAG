@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul

echo [PersonalityRAG] Windows launcher

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "RUNTIME_LOCK=requirements-runtime.lock"
set "DEPENDENCY_MARKER=.venv\.personalityrag-dependencies.json"

if exist "%VENV_PYTHON%" (
  "%VENV_PYTHON%" tools\runtime_bootstrap.py validate
  if errorlevel 1 (
    echo [PersonalityRAG] Existing .venv is not CPython 3.10+ x64.
    echo [PersonalityRAG] The launcher will not delete or overwrite it. Preserve or rename it, then create a compatible environment.
    pause
    exit /b 1
  )
) else (
  set "PYTHON_SPEC="
  call :select_python -3.12
  call :select_python -3
  call :select_python -3.11
  call :select_python -3.10
  if not defined PYTHON_SPEC (
    echo [PersonalityRAG] CPython 3.10+ x64 was not found. Please install a compatible version first.
    pause
    exit /b 1
  )
  echo [PersonalityRAG] Creating a virtual environment with !PYTHON_SPEC!...
  py !PYTHON_SPEC! -m venv .venv
  if errorlevel 1 (
    echo [PersonalityRAG] Compatible Python virtual environment creation failed.
    pause
    exit /b 1
  )
  "%VENV_PYTHON%" tools\runtime_bootstrap.py validate
  if errorlevel 1 (
    echo [PersonalityRAG] The new virtual environment failed CPython 3.10+ x64 validation.
    pause
    exit /b 1
  )
)

if not exist "%RUNTIME_LOCK%" (
  echo [PersonalityRAG] Missing %RUNTIME_LOCK%. Runtime dependencies cannot be verified.
  pause
  exit /b 1
)

set "PERSONALITYRAG_RECOVERY_STATE_ROOT=%PERSONALITYRAG_STATE_ROOT%"
if not defined PERSONALITYRAG_RECOVERY_STATE_ROOT set "PERSONALITYRAG_RECOVERY_STATE_ROOT=%CD%"
"%VENV_PYTHON%" tools\update_helper.py recover "%PERSONALITYRAG_RECOVERY_STATE_ROOT%" --active-transaction "%PERSONALITYRAG_UPDATE_TRANSACTION%"
if errorlevel 1 (
  echo [PersonalityRAG] Update transaction recovery failed. Startup was stopped to protect the installation.
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

for /f "usebackq delims=" %%V in (`"%VENV_PYTHON%" -c "from personalityrag.version import display_version; print(display_version())" 2^>nul`) do set "PERSONALITYRAG_VERSION=%%V"
if not defined PERSONALITYRAG_VERSION set "PERSONALITYRAG_VERSION=unknown"
echo [PersonalityRAG] Starting WebUI %PERSONALITYRAG_VERSION%...
"%VENV_PYTHON%" run.py
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:select_python
if defined PYTHON_SPEC exit /b 0
py %~1 -c "import struct,sys;raise SystemExit(0 if sys.implementation.name=='cpython' and sys.version_info[:2]>=(3,10) and struct.calcsize('P')==8 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_SPEC=%~1"
exit /b 0
