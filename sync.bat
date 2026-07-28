@echo off
chcp 65001 > nul
cd /d "%~dp0"
call git-config.bat

if not exist .git (
    echo .git not found. Clone the repository first.
    pause
    exit /b 1
)

set REMOTE_URL=https://%GITHUB_USER%:%GITHUB_PAT%@github.com/%GITHUB_REPO%.git
set CLEAN_URL=https://github.com/%GITHUB_REPO%.git

git remote set-url origin "%REMOTE_URL%" 2> nul

echo === git fetch --all --prune ===
git -c credential.helper= fetch --all --prune
if errorlevel 1 (
    git remote set-url origin "%CLEAN_URL%"
    echo Failed to fetch.
    pause
    exit /b 1
)

echo.
echo === git pull ===
git -c credential.helper= pull
if errorlevel 1 (
    git remote set-url origin "%CLEAN_URL%"
    echo Failed to pull.
    pause
    exit /b 1
)

git remote set-url origin "%CLEAN_URL%"

echo.
echo Sync complete.
pause
exit /b 0
