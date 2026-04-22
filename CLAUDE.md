# ImageAnalyzer

Local image analysis API server that performs OCR, character recognition, face recognition, costume recognition, and full-body person recognition. Called from a Chrome extension.

## Tech Stack
- **Training**: Python (ultralytics YOLOv8, torch, onnx)
- **Server**: Node.js + TypeScript (Express, onnxruntime-node, sharp, better-sqlite3)
- **OCR**: PaddleOCR (DBNet detection + CRNN recognition) + manga-ocr (Japanese fallback), all ONNX
- **Scripts**: TypeScript (ts-node)

## Key Commands
```bash
npm run server              # Start analysis server (port 3000)
npm run dev                 # Dev mode with auto-reload
npm run register:characters # Register character embeddings
npm run register:faces      # Register face embeddings
npm run register:costumes   # Register costume embeddings
npm run register:persons    # Register full-body person embeddings
npm run register:all        # Register all of the above
```

## Architecture
- YOLO detects character/face/person regions → CLIP/ArcFace extracts embeddings → cosine similarity matching against SQLite DB
- OCR via PaddleOCR (Chinese CRNN as primary, Korean/English fallbacks) + manga-ocr (Japanese fallback), matched against `searchStrings.tsv`
- All ONNX models loaded once at startup and kept in memory
- CLIP, ArcFace, face-detection, YOLO-person (COCO), PaddleOCR, manga-ocr models auto-downloaded on first run
- YOLO character model requires separate training

## Important Files
- `config.json` — similarity thresholds (character/face/costume/person)
- `searchStrings.tsv` — OCR search strings (tab-separated for composite conditions)
- `server/src/index.ts` — main server entry point, POST /analyze endpoint
- `server/src/ocr.ts` — OCR pipeline (PaddleOCR + manga-ocr)
- `server/src/inference.ts` — all ONNX model inference logic (CLIP, ArcFace, YOLO, face detection)
- `server/src/db.ts` — SQLite schema and queries
- `server/src/model-downloader.ts` — auto-download logic for ONNX models
- `Document.md` — detailed per-file class/method documentation, workflow, config structure. **Must be updated alongside any code changes.**
