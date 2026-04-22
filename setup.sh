#!/bin/bash
set -e

echo "=== ImageAnalyzer Setup ==="

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

# Create directories
echo "Creating directories..."
mkdir -p data/characters data/faces data/costumes data/persons data/yolo-training/images data/yolo-training/labels
mkdir -p models db

# Download ONNX models (CLIP, ArcFace, face detection)
echo "Downloading ONNX models..."
npx ts-node -e "
const { ensureModelsDownloaded } = require('./server/src/model-downloader');
ensureModelsDownloaded('./models').then(() => console.log('Model download complete.'));
"

# Initialize database
echo "Initializing database..."
npx ts-node -e "
const { initDB } = require('./server/src/db');
initDB('./db');
console.log('Database initialized.');
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  1. Training (Python):"
echo "     cd training && pip install -r requirements.txt"
echo "     python augment.py    # Generate augmented training data"
echo "     python train.py      # Train YOLO model"
echo ""
echo "  2. Register embeddings:"
echo "     npm run register:characters  # Register character embeddings"
echo "     npm run register:faces       # Register face embeddings"
echo "     npm run register:all         # Register both"
echo ""
echo "  3. Start server:"
echo "     npm run server   # Production mode"
echo "     npm run dev      # Development mode (auto-reload)"
echo ""
echo "  4. API endpoint:"
echo "     POST http://localhost:3000/analyze (multipart/form-data, field: image)"
