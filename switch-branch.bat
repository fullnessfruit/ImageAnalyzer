@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
cd /d "%~dp0"
call git-config.bat

set BRANCH_NAME=

if "%BRANCH_NAME%"=="" (
    echo BRANCH_NAME is not set. Edit switch-branch.bat and set the branch name.
    pause
    exit /b 1
)

if not exist .git (
    echo .git not found. Clone the repository first.
    pause
    exit /b 1
)

set REMOTE_URL=https://%GITHUB_USER%:%GITHUB_PAT%@github.com/%GITHUB_REPO%.git
set CLEAN_URL=https://github.com/%GITHUB_REPO%.git

git remote set-url origin "%REMOTE_URL%" 2> nul

echo === git fetch origin --prune ===
git -c credential.helper= fetch origin --prune
if errorlevel 1 (
    git remote set-url origin "%CLEAN_URL%"
    echo Failed to fetch.
    pause
    exit /b 1
)

git remote set-url origin "%CLEAN_URL%"

rem Local exists?
git show-ref --verify --quiet "refs/heads/%BRANCH_NAME%"
if not errorlevel 1 goto local_exists

rem Remote exists?
git show-ref --verify --quiet "refs/remotes/origin/%BRANCH_NAME%"
if not errorlevel 1 goto remote_only

rem Nowhere - create new
echo Branch "%BRANCH_NAME%" does not exist locally or on remote. Creating new branch.
git checkout -b "%BRANCH_NAME%"
goto done

:remote_only
echo Branch "%BRANCH_NAME%" exists on remote only. Switching ^(auto-tracks origin/%BRANCH_NAME%^).
git switch "%BRANCH_NAME%"
goto done

:local_exists
echo Branch "%BRANCH_NAME%" exists locally. Already fetched. Switching.
git switch "%BRANCH_NAME%"
goto done

:done
if errorlevel 1 (
    echo Failed to switch.
    pause
    exit /b 1
)
echo Done. Current branch:
git rev-parse --abbrev-ref HEAD
pause
exit /b 0
