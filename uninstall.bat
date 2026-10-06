@echo off
REM Remove installation artifacts while preserving shared credentials and user data.
setlocal
echo === ImageAnalyzer Uninstall ===
echo Keeping shared OCR_BROKER_SECRET for VoiceAnalyzer and other clients.
echo To delete ONLY the shared key, run: "%~dp0delete-shared-secret.bat"
echo Doing so requires reconfiguring both analyzers and their clients together.
node "%~dp0scripts\uninstall.cjs"
exit /b %errorlevel%
