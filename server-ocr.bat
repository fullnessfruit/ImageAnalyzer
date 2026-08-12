@echo off
REM Start the analysis server with OCR only.
REM
REM Vision models (face / arcface / anime detectors / ccip / wd-tagger, ~772 MB) are neither
REM downloaded nor loaded, so the process fits on a small host. faces/characters/costumes
REM come back empty; ocr is unaffected.
REM
REM Use this when the caller only needs "does this image contain one of these strings?".

cd /d "%~dp0"
call npm run server:ocr
