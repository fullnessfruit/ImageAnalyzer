# ImageAnalyzer

Local image analysis API server called from a Chrome extension. Answers four questions about an image:
OCR list matching, real-person face identification, anime character identification, anime costume
identification. Full-text extraction, scene understanding, silhouette identification, and real-person
full-body ReID are explicitly out of scope.

## Tech Stack
- **Server**: Node.js + TypeScript (Express, onnxruntime-node, sharp, better-sqlite3). No Python.
- **OCR**: PaddleOCR DBNet detection + multilingual CRNN recognition, ONNX
- **Vision**: InsightFace SCRFD + ArcFace (real faces), deepghs YOLOv8 detectors + CCIP (anime),
  SmilingWolf WD-Tagger v3 (character names + clothing tags)

## Key Commands
```bash
npm run server                       # Start analysis server (port 3000)
npm run dev                          # Dev mode with auto-reload
npm run register:all                 # Register face + character + costume galleries
npm run analyze -- sample/           # Analyze file/folder without the server (same code path)
npm run tags -- sample/image.jpg     # Dump WD-Tagger raw probabilities
npm run detect -- sample/image.jpg   # Dump detection boxes and face landmarks
npm run search-strings -- --list     # Manage searchStrings.tsv
```

## Architecture
- **OCR is keyword spotting, not reading.** Search strings are scored directly against the CTC
  probability lattice; there is no full-text assembly and no substring comparison. Per-position
  alternative character sets absorb kana/width/case variation and OCR confusions.
- The search string's script picks the recognition model — the `ch` dictionary covers kanji/kana/latin,
  the `ko` dictionary covers hangul only. No language detection on the image.
- **CLIP is not used.** Its space clusters by art style, not identity.
- Real faces require 5-point landmark alignment before ArcFace; unaligned crops are out of distribution.
- Costumes are IDF-weighted clothing-tag vectors with the head masked, so matching is invariant to art
  style and to who is wearing the outfit.
- Thresholds are per-model and not mutually comparable; matches must also clear a top-1 vs top-2 margin.
- All ONNX models load once at startup and are auto-downloaded on first run. No model training required.

## Important Files
- `config.json` — thresholds, margins, OCR settings, character name aliases
- `searchStrings.tsv` — one list per line; tab-separated parts are AND, lines are OR
- `server/src/ocr.ts` — detection, strip building, batched recognition, CTC keyword scoring
- `server/src/inference.ts` — all ONNX inference (SCRFD, ArcFace, anime detectors, CCIP, WD-Tagger)
- `server/src/analyze.ts` — orchestration shared by the server and the CLI
- `server/src/matching.ts` — centroid + max scoring with margin rejection
- `server/src/db.ts` — SQLite; `kind` × `space` keeps embedding spaces from being compared
- `server/src/model-downloader.ts` — HuggingFace redirect handling
- `server/src/index.ts` — HTTP layer only
- `Document.md` — per-file intent, logic, and architectural decisions. **Must be updated alongside any
  code changes.**
