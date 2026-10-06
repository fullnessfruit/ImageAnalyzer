@echo off
REM Explicitly delete only the shared authentication key, not services or data.
setlocal
echo This key is shared by ImageAnalyzer, VoiceAnalyzer, and the OCR broker.
node "%~dp0scripts\shared-secret.cjs" delete
if errorlevel 1 exit /b 1
echo Key deletion complete. Existing clients need the same replacement key.
pause
