@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
cd /d "%~dp0"

if not exist .git (
    echo .git not found. Clone the repository first.
    pause
    exit /b 1
)

echo === Local branches ===
git branch
echo.

set TARGET=
set /p TARGET="Enter branch name to delete: "

if "%TARGET%"=="" (
    echo No branch specified. Aborted.
    pause
    exit /b 1
)

git branch -d "%TARGET%"
if errorlevel 1 (
    echo Failed to delete branch "%TARGET%".
    pause
    exit /b 1
)

echo Branch "%TARGET%" deleted.
pause
exit /b 0
