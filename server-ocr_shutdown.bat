@echo off
REM Start the OCR worker and shut the PC down after a fixed time.
REM
REM Same worker as server-ocr.bat (this script calls it, so the env defaults and the
REM explanations live there and are not duplicated here). The only addition is a timed
REM shutdown: set SHUTDOWN_AFTER_SEC below and the machine powers off that many seconds
REM after this script starts, whether or not the worker is still busy. A job cut off
REM mid-OCR is not lost - its broker lease expires and another run picks it up.
REM
REM HOW THE TIMER WORKS
REM   The shutdown is registered with Windows up front (shutdown /s /t N), not counted down
REM   by this script. That makes it visible (Windows shows the pending shutdown) and
REM   cancellable with the standard command:
REM
REM     shutdown /a
REM
REM   It also means the timer keeps running if you close this window or the worker dies.
REM   If you do not want the PC to turn off any more, run shutdown /a.
REM
REM   Windows allows only ONE pending shutdown. If one is already registered this script
REM   refuses to start rather than run the worker with a timer you did not choose - cancel
REM   the existing one with shutdown /a and start again. The worker's low-battery shutdown
REM   still works while the timer is pending: it replaces the timer with its own short
REM   countdown, because a draining battery is the more urgent of the two.
REM
REM   The value is capped by Windows at 315360000 seconds (10 years).

REM Seconds from now until shutdown. 3600 = 1 hour.
set SHUTDOWN_AFTER_SEC=7200

shutdown /s /t %SHUTDOWN_AFTER_SEC% /c "ImageAnalyzer OCR worker: timed shutdown (%SHUTDOWN_AFTER_SEC% s). Cancel with: shutdown /a"
if errorlevel 1 (
  echo Refusing to start - could not schedule the shutdown. A shutdown may already be pending; run "shutdown /a" and start again.
  exit /b 1
)
echo Shutdown scheduled - seconds: %SHUTDOWN_AFTER_SEC% (cancel: shutdown /a)

call "%~dp0server-ocr.bat"
