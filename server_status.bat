@echo off
REM Follow the running worker's audit log in a visible window without restarting it.
cd /d "%~dp0"
if errorlevel 1 exit /b 1
title ImageAnalyzer worker status
node scripts\worker-status.cjs --watch
if errorlevel 1 pause
