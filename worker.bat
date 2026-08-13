@echo off
REM Start the OCR worker: pull jobs from the broker, analyze, post results back.
REM
REM Use this on the machine that is BEHIND NAT. The Chrome extension cannot call in, so the
REM direction is reversed and this process does the pulling. OCR only - the six vision models
REM (~772 MB) are neither downloaded nor loaded, and the gallery DB is never opened.
REM
REM The broker runs next to the extension (AnnouncementAggregator\ocr-broker). It listens on
REM all interfaces and the firewall restricts access to this laptop's IP, so normally just
REM point at it:
REM
REM   set OCR_BROKER_URL=http://<broker-machine>:3100
REM
REM OCR_BROKER_SECRET must match the broker and the extension popup.
REM
REM The secret is never sent - each request carries only an HMAC-SHA256 signature over itself
REM (scheme OCR1), and the broker signs its responses back so an impostor holding the port is
REM detected. Both machines need their clocks in sync (NTP, on by default): signatures carry
REM a timestamp and are rejected more than 2 minutes out.
REM
REM Env: OCR_BROKER_URL, OCR_BROKER_SECRET, OCR_WORKER_INTERVAL_MS (60000), OCR_WORKER_ID

cd /d "%~dp0"

if "%OCR_BROKER_URL%"=="" set OCR_BROKER_URL=http://localhost:3100

call npm run worker
