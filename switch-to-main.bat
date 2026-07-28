@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist .git (
    echo .git not found. Clone the repository first.
    pause
    exit /b 1
)

git switch main
if errorlevel 1 (
    echo Failed to switch to main branch.
    pause
    exit /b 1
)

echo Switched to main branch.
pause
exit /b 0
