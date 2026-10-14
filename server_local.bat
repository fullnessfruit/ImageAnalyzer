@echo off
setlocal
REM Use the same local broker port as the extension and VoiceAnalyzer.
set "OCR_BROKER_URL=http://127.0.0.1:3100"
call "%~dp0server.bat"
exit /b %errorlevel%
