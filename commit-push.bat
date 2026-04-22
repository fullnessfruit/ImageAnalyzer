@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
cd /d "%~dp0"
call git-config.bat

set REMOTE_URL=https://%GITHUB_USER%:%GITHUB_PAT%@github.com/%GITHUB_REPO%.git

if not exist .git (
    echo .git not found. Clone the repository first.
    pause
    exit /b 1
)

echo === Changed files ===
git status -s
echo.

for /f "delims=" %%i in ('git status -s') do goto has_changes
echo No changes to commit. Pushing only.
goto do_push

:has_changes
for /f "delims=" %%f in ('git ls-files --others --exclude-standard') do call :ask_untracked "%%f"
for /f "delims=" %%f in ('git diff --name-only') do call :review_modified "%%f"

echo.
echo === Staged ===
git diff --cached --stat
echo.

git diff --cached --quiet 2> nul
if not errorlevel 1 (
    echo Nothing staged. Pushing only.
    goto do_push
)

call commit-msg.bat

if "%COMMIT_MSG%"=="" (
    echo No commit message set. Edit commit-msg.bat first.
    pause
    exit /b 1
)

echo Commit message: %COMMIT_MSG%
set /p CONFIRM="Commit and push? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Aborted.
    pause
    exit /b 0
)

git -c user.name="%GIT_USER_NAME%" -c user.email="%GIT_USER_EMAIL%" commit -m "%COMMIT_MSG%"

:do_push
echo.
set CLEAN_URL=https://github.com/%GITHUB_REPO%.git
git remote set-url origin "%REMOTE_URL%" 2> nul
rem Disable all credential helpers for this push so the PAT is not saved to Windows Credential Manager
git -c credential.helper= push -u origin main
git remote set-url origin "%CLEAN_URL%"
pause
exit /b 0

:ask_untracked
set /p ADD_FILE="Add untracked file '%~1'? (y/n): "
if /i "!ADD_FILE!"=="y" git add "%~1"
goto :eof

:review_modified
git check-ignore --no-index -q "%~1"
if not errorlevel 1 (
    echo === %~1 [gitignored - skipped] ===
    goto :eof
)
echo.
echo === %~1 ===
git diff "%~1"
echo.
set /p STAGE_FILE="Stage '%~1'? (y/n): "
if /i "!STAGE_FILE!"=="y" git add "%~1"
goto :eof
