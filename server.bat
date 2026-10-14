@echo off
REM Start the full broker worker: OCR, real faces, anime characters and costumes.
REM Run setup.bat and register reference images before relying on gallery matches.
cd /d "%~dp0"
if errorlevel 1 exit /b 1

REM An inherited OCR-only setting must not silently disable the vision models.
set "IMAGEANALYZER_OCR_ONLY=0"
if "%OCR_BROKER_URL%"=="" set "OCR_BROKER_URL=http://ec2-35-78-169-107.ap-northeast-1.compute.amazonaws.com:3100"
if "%OCR_WORKER_BATTERY_SHUTDOWN_PERCENT%"=="" set "OCR_WORKER_BATTERY_SHUTDOWN_PERCENT=25"
call npm run worker -- --all
exit /b %errorlevel%
