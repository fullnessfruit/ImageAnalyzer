@echo off
echo === ImageAnalyzer Setup ===

echo Installing Node.js dependencies...
call npm install
if errorlevel 1 goto :error

echo Creating directories...
if not exist "data\faces" mkdir "data\faces"
if not exist "data\characters" mkdir "data\characters"
if not exist "data\costumes" mkdir "data\costumes"
if not exist "models" mkdir "models"
if not exist "db" mkdir "db"

echo Downloading ONNX models (about 890 MB on first run)...
call npx ts-node -e "require('./server/src/model-downloader').ensureModelsDownloaded('./models').then(() => console.log('Model download complete.'));"
if errorlevel 1 goto :error

echo Initializing database...
call npx ts-node -e "require('./server/src/db').initDB('./db'); console.log('Database initialized.');"
if errorlevel 1 goto :error

echo.
echo === Setup Complete ===
echo.
echo Usage:
echo   1. Place reference images (folder name = the name reported by the analyzer):
echo        data\faces\^<person-name^>\          real photos, no cropping needed
echo        data\characters\^<character-name^>\  anime character images
echo        data\costumes\^<costume-name^>\      costume images
echo.
echo   2. Register:
echo        npm run register:all
echo.
echo   3. Set OCR search lists in searchStrings.tsv (one list per line, tab = AND)
echo.
echo   4. Start server:
echo        npm run server   # production
echo        npm run dev      # auto-reload
echo.
echo   5. Analyze without the server (threshold tuning, regression checks):
echo        npm run analyze -- sample\
goto :eof

:error
echo.
echo === Setup failed ===
exit /b 1
