@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
cd /d "%~dp0"
call git-config.bat

if not exist .git (
    echo .git not found. Clone the repository first.
    pause
    exit /b 1
)

echo === Changed files ===
git status -s
echo.

for /f "delims=" %%i in ('git status -s') do goto has_changes
echo No changes.
pause
exit /b 0

:has_changes
for /f "delims=" %%f in ('git ls-files --others --exclude-standard') do call :ask_untracked "%%f"
for /f "delims=" %%f in ('git diff --name-only') do call :review_modified "%%f"

echo.
echo === Staged ===
git diff --cached --stat
echo.

call commit-msg.bat

if "%COMMIT_MSG%"=="" (
    echo No commit message set. Edit commit-msg.bat first.
    pause
    exit /b 1
)

echo Commit message: %COMMIT_MSG%
set /p CONFIRM="Commit? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Aborted.
    pause
    exit /b 0
)

git -c user.name="%GIT_USER_NAME%" -c user.email="%GIT_USER_EMAIL%" commit -m "%COMMIT_MSG%"
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
if /i "%~1"=="ChannelList.js" (
    echo [View only - skipped]
) else (
    set /p STAGE_FILE="Stage '%~1'? (y/n): "
    if /i "!STAGE_FILE!"=="y" git add "%~1"
)
goto :eof
