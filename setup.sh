#!/bin/bash
set -e

echo "=== ImageAnalyzer Setup ==="

echo "Installing Node.js dependencies..."
npm install

echo "Creating directories..."
mkdir -p data/faces data/characters data/costumes models db

echo "Downloading ONNX models (about 890 MB on first run)..."
npx ts-node -e "require('./server/src/model-downloader').ensureModelsDownloaded('./models').then(() => console.log('Model download complete.'))"

echo "Initializing database..."
npx ts-node -e "require('./server/src/db').initDB('./db')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  1. Place reference images (folder name = the name reported by the analyzer):"
echo "       data/faces/<person-name>/         real photos, no cropping needed"
echo "       data/characters/<character-name>/ anime character images"
echo "       data/costumes/<costume-name>/     costume images"
echo ""
echo "  2. Register:"
echo "       npm run register:all"
echo ""
echo "  3. Set OCR search lists in searchStrings.tsv (one list per line, tab = AND)"
echo ""
echo "  4. Start server:"
echo "       npm run server    # production"
echo "       npm run dev       # auto-reload"
echo ""
echo "  5. Analyze without the server (threshold tuning, regression checks):"
echo "       npm run analyze -- sample/"
