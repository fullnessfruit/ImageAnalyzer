@echo off
REM Start the full broker worker and shut Windows down after a fixed time.
REM The timer is registered with Windows, so it remains active if this window closes.
REM Cancel a pending shutdown with: shutdown /a

REM Seconds from now until shutdown. 7200 = 2 hours.
set "SHUTDOWN_AFTER_SEC=7200"

shutdown /s /t %SHUTDOWN_AFTER_SEC% /c "ImageAnalyzer server: timed shutdown (%SHUTDOWN_AFTER_SEC% s). Cancel with: shutdown /a"
if errorlevel 1 (
  echo Refusing to start - could not schedule the shutdown. A shutdown may already be pending; run "shutdown /a" and start again.
  exit /b 1
)
echo Shutdown scheduled - seconds: %SHUTDOWN_AFTER_SEC% (cancel: shutdown /a)

call "%~dp0server.bat"
exit /b %errorlevel%
