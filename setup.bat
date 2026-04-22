@echo off
echo === ImageAnalyzer Setup ===

echo Installing Node.js dependencies...
call npm install
if errorlevel 1 goto :error

echo Creating directories...
if not exist "data\characters" mkdir "data\characters"
if not exist "data\faces" mkdir "data\faces"
if not exist "data\costumes" mkdir "data\costumes"
if not exist "data\persons" mkdir "data\persons"
if not exist "data\yolo-training\images" mkdir "data\yolo-training\images"
if not exist "data\yolo-training\labels" mkdir "data\yolo-training\labels"
if not exist "models" mkdir "models"
if not exist "db" mkdir "db"

echo Downloading ONNX models...
call npx ts-node -e "const { ensureModelsDownloaded } = require('./server/src/model-downloader'); ensureModelsDownloaded('./models').then(() => console.log('Model download complete.'));"
if errorlevel 1 goto :error

echo Initializing database...
call npx ts-node -e "const { initDB } = require('./server/src/db'); initDB('./db'); console.log('Database initialized.');"
if errorlevel 1 goto :error

echo.
echo === Setup Complete ===
echo.
echo Usage:
echo   1. Training (Python):
echo      cd training ^& pip install -r requirements.txt
echo      python augment.py    # Generate augmented training data
echo      python train.py      # Train YOLO model
echo.
echo   2. Register embeddings:
echo      npm run register:characters  # Register character embeddings
echo      npm run register:faces       # Register face embeddings
echo      npm run register:all         # Register both
echo.
echo   3. Start server:
echo      npm run server   # Production mode
echo      npm run dev      # Development mode (auto-reload)
echo.
echo   4. API endpoint:
echo      POST http://localhost:3000/analyze (multipart/form-data, field: image)
goto :eof

:error
echo.
echo === Setup failed ===
exit /b 1
