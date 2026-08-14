@echo off
REM Start the OCR worker: pull jobs from the broker, analyze, post results back.
REM
REM THIS IS THE ONLY SCRIPT TO RUN ON THIS MACHINE.
REM
REM The name is inherited from the script this replaced, which started the HTTP analysis
REM server in OCR-only mode. Nothing calls that server any more - the extension reaches OCR
REM through the broker instead - so the two scripts collapsed into one and the familiar name
REM won. `npm run server:ocr` still exists for the manual curl API documented in README; it
REM simply has no launcher of its own now.
REM
REM Strictly speaking this is a client, not a server: it listens on nothing and polls the
REM broker. That is also why there is no address to bind here - only OCR_BROKER_URL, the
REM address it calls OUT to.
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
REM LOW-BATTERY SHUTDOWN
REM   This runs unattended on a laptop and burns CPU on OCR, so draining to a hard power cut
REM   mid-write is a real way to lose data. A controlled shutdown is better. The code default
REM   is OFF; this script is what arms it.
REM   It counts only while DISCHARGING - on AC it never fires, whatever the percentage - and
REM   it schedules the shutdown with a grace period rather than pulling the plug, so the job
REM   in flight has time to upload its result. To cancel during the grace period:  shutdown /a
REM
REM Env: OCR_BROKER_URL, OCR_BROKER_SECRET, OCR_WORKER_INTERVAL_MS (60000), OCR_WORKER_ID,
REM OCR_WORKER_BATTERY_SHUTDOWN_PERCENT (0 = off), OCR_WORKER_BATTERY_CHECK_MS (60000),
REM OCR_WORKER_BATTERY_GRACE_SEC (120)

cd /d "%~dp0"

if "%OCR_BROKER_URL%"=="" set OCR_BROKER_URL=http://localhost:3100

REM Shut down at 25 percent while on battery. Set the variable to 0 to disable.
if "%OCR_WORKER_BATTERY_SHUTDOWN_PERCENT%"=="" set OCR_WORKER_BATTERY_SHUTDOWN_PERCENT=25

call npm run worker
