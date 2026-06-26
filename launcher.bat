@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo [PersonalityRAG] Windows v0.1.0 launcher

if not exist ".venv\Scripts\python.exe" (
  echo [PersonalityRAG] 正在创建 Python 3.12 虚拟环境...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo [PersonalityRAG] 未找到 Python 3.12。请先安装 Python 3.12 x64。
    pause
    exit /b 1
  )
)

echo [PersonalityRAG] 正在检查依赖...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo [PersonalityRAG] 依赖安装失败。
  pause
  exit /b 1
)

echo [PersonalityRAG] 正在启动 WebUI...
".venv\Scripts\python.exe" run.py
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
