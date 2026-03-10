@echo off
setlocal

REM Ensure this script runs from the repository root.
cd /d "%~dp0"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Current folder is not a Git repository.
  exit /b 1
)

if not exist ".gitmodules" (
  echo [ERROR] .gitmodules not found. No submodule to update.
  exit /b 1
)

set "SUBMODULE_PATH=%~1"

echo Syncing submodule config...
git submodule sync --recursive
if errorlevel 1 (
  echo [ERROR] Failed to sync submodule config.
  exit /b 1
)

if "%SUBMODULE_PATH%"=="" (
  echo Updating all submodules to latest remote commit...
  git submodule update --init --recursive --remote
) else (
  echo Updating submodule "%SUBMODULE_PATH%" to latest remote commit...
  git submodule update --init --remote -- "%SUBMODULE_PATH%"
)

if errorlevel 1 (
  echo [ERROR] Submodule update failed.
  exit /b 1
)

echo Update completed successfully.
exit /b 0
