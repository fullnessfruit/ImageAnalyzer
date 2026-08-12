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
setup.bat                            # First run: deps, dirs, ONNX download (~890 MB). ./setup.sh on POSIX
npm run server                       # Start analysis server (port 3000)
npm run server:ocr                   # Start with OCR only (no vision models). server-ocr.bat on Windows
npm run dev                          # Dev mode with auto-reload
npm run register:all                 # Register face + character + costume galleries
npm run candidates -- --list         # Review collected crops that are waiting for approval
npm run analyze -- sample/           # Analyze file/folder without the server (same code path)
npm run tags -- sample/image.jpg     # Dump WD-Tagger raw probabilities
npm run detect -- sample/image.jpg   # Dump detection boxes and face landmarks
npm run search-strings -- --list     # Manage searchStrings.tsv
```
`register:faces` / `register:characters` / `register:costumes` register a single kind. `candidates` also
takes `--promote <kind> <name>` and `--clear [kind] [name]`, where kind is `face`, `character`, or
`costume`.

## Architecture
- **The caller owns the search list.** `POST /analyze` takes it as the multipart field
  `searchStrings` (TSV text); `searchStrings.tsv` on disk is only the fallback for manual runs.
  A caller whose keywords change (the Chrome extension) must send them, otherwise "not in the
  list" and "in the list but unreadable" are indistinguishable in the answer.
- **OCR-only mode** (`--ocr-only` / `IMAGEANALYZER_OCR_ONLY=1`) skips downloading and loading
  the six vision models. Detectors are null-safe, so `analyzeImage` still runs and simply
  returns empty faces/characters/costumes.
- **OCR is keyword spotting, not reading.** Search strings are scored directly against the CTC
  probability lattice; there is no full-text assembly and no substring comparison. Per-position
  alternative character sets absorb kana/width/case variation and OCR confusions.
- The search string's script picks the recognition model - the `ch` dictionary covers kanji/kana/latin,
  the `ko` dictionary covers hangul only. No language detection on the image.
- **CLIP is not used.** Its space clusters by art style, not identity.
- Real faces require 5-point landmark alignment before ArcFace; unaligned crops are out of distribution.
- **Faces are reported in two bands.** `faces` clears the `face` threshold and is treated as a live
  subject; `facesWeak` falls between `faceWeak` and `face`, which is where photos-of-photos and
  low-quality crops land. Never auto-decide on `facesWeak`.
- **Characters are reported only when already registered.** CCIP is the identity axis and matches from
  a single reference image. WD-Tagger is an auxiliary signal whose fixed vocabulary does not cover
  recent characters, so tagger names pass only if the gallery already holds them.
- Costumes are IDF-weighted clothing-tag vectors with the head masked, so matching is invariant to art
  style and to who is wearing the outfit.
- Thresholds are per-model and not mutually comparable; matches must also clear a top-1 vs top-2 margin.
- **Nothing is auto-registered.** Confirmed crops are written to `data/_candidates/<kind>/<name>/` for
  human review; a gallery changes only after `candidates --promote` followed by a `register` run.
  Auto-registration would let a wrong entry silently alter later matches with no way to see which entry
  caused it.
- All ONNX models load once at startup and are auto-downloaded on first run. No model training required.

## Important Files
- `config.json`: thresholds, margins, OCR settings, character name aliases, candidate collection limits
- `searchStrings.tsv`: one list per line; tab-separated parts are AND, lines are OR
- `server/src/ocr.ts`: detection, strip building, batched recognition, CTC keyword scoring
- `server/src/inference.ts`: all ONNX inference (SCRFD, ArcFace, anime detectors, CCIP, WD-Tagger)
- `server/src/analyze.ts`: orchestration shared by the server and the CLI
- `server/src/matching.ts`: centroid + max scoring with margin rejection
- `server/src/candidates.ts`: candidate crop collection, dedup, and per-name caps
- `server/src/db.ts`: SQLite; `embeddings` is the approved gallery and `candidates` is the pending
  review queue; `kind` × `space` keeps embedding spaces from being compared
- `server/src/model-downloader.ts`: HuggingFace redirect handling
- `server/src/index.ts`: HTTP layer only

## Detailed Documentation

`Document.md` - the document that records the **intent**, **logic**, **system description**, and
**important architectural decisions** of every file, class, and function.
Before beginning any work for the first time, read all of `Document.md` first.
Before beginning work, always read `## Document Editing Principles` and `## Programming Work Principles`
in `Document.md` and strictly comply with them.
The Document Editing Principles and Programming Work Principles in that document are principles that
must be followed, but every other part is never a description of constraint specifications that the
current code must satisfy. It is documentation reflecting the content of the code so that AI can
understand the code quickly, and whenever the content of the code changes it must always be revised to
reflect the latest state of the code.
