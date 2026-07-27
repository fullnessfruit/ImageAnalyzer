# ImageAnalyzer

Local API server for image analysis. Accepts an image and answers four questions, designed to be
called from a Chrome extension:

1. **OCR list matching** — does the image contain any of the strings in `searchStrings.tsv`?
2. **Real-person face** — is a registered person's face present?
3. **Anime character** — which character is this?
4. **Anime costume** — is a registered costume being worn?

It does not extract full text, describe scenes, identify silhouettes, or do real-person full-body
re-identification. Those are out of scope by design.

## How it works

**OCR is treated as keyword spotting, not reading.** The search list is known in advance, so instead
of decoding text and comparing substrings, each candidate string is scored directly against the CTC
probability lattice. A greedy decode picks the top class at every timestep and its exact-match
probability decays as the L-th power of per-character accuracy; lattice scoring survives a character
being pushed to second place. Observed: a region misread as `大西亚玖璃` (simplified 亚) still scored
0.83 for `大西亜玖璃`, and `虹ヶ咲` misread as `虹久咲` scored 0.93 once `ヶ`/`久` confusion was modelled.

**Identity matching uses task-appropriate embedding spaces.** CLIP is not used anywhere — its space
clusters by art style and composition rather than identity, so the same character in a different style
lands far away while different characters with similar attributes land close.

| Task | Model | Registration needed |
|---|---|---|
| Real-person face | SCRFD landmarks → 5-point similarity warp → ArcFace w600k_r50 (flip TTA) | 1–3 photos per person |
| Anime character | WD-Tagger v3 (zero-shot names) + CCIP (gallery) | 0 for known characters, 1–2 otherwise |
| Anime costume | WD-Tagger clothing tags, IDF-weighted, head masked | 1 image per costume |

ArcFace is trained only on faces warped to a canonical 5-point template, so landmark alignment is
mandatory — feeding a raw bounding-box crop puts every input out of distribution.

Costumes are represented as a clothing-tag probability vector rather than an embedding. Tags are
semantic, so the representation is invariant to art style; they describe garments, so it is invariant
to who is wearing them. The head is masked before tagging to remove identity cues.

## Folder structure

```
ImageAnalyzer/
├── data/
│   ├── faces/<person-name>/         # photos for face registration
│   ├── characters/<character-name>/ # images for CCIP registration
│   └── costumes/<costume-name>/     # images for costume registration
├── models/                          # ONNX models (auto-downloaded)
├── db/embeddings.db                 # SQLite gallery
├── server/src/                      # analysis server
├── scripts/                         # registration + diagnostic CLIs
├── config.json                      # thresholds, margins, aliases
└── searchStrings.tsv                # OCR search lists
```

Folder name = the name reported by the analyzer. Faces do not need cropping — detection handles it.

## Quick start

```bash
npm install
npm run register:all
npm run server
```

Models (~890 MB) download automatically on first run.

### API

```bash
curl -X POST http://localhost:3000/analyze -F "image=@test.jpg"
```

```json
{
  "ocr": {
    "found": ["大西亜玖璃"],
    "detail": [{ "list": "大西亜玖璃", "matched": true, "parts": [{ "text": "大西亜玖璃", "score": 0.834 }] }],
    "regions": 19,
    "fullText": "debug greedy decode, not used for matching"
  },
  "faces":      [{ "name": "大西亜玖璃", "score": 0.6394, "margin": 0.6394, "box": [x, y, w, h] }],
  "characters": [{ "name": "上原歩夢", "score": 0.9639, "margin": 0, "source": "tagger" }],
  "costumes":   [{ "name": "虹ヶ咲制服", "score": 1.0, "margin": 1.0, "source": "blazer,skirt,..." }],
  "_detections": { "realFaces": 2, "animeFaces": 2, "animePersons": 2, "textRegions": 19 },
  "_elapsedMs": 30607
}
```

`GET /health` reports which models loaded and how many gallery entries exist per task.

## Configuration

`config.json` and `searchStrings.tsv` are re-read on every request — no restart needed.

```json
{
  "similarityThreshold": { "face": 0.28, "character": 0.82, "costume": 0.55 },
  "margin":              { "face": 0.06, "character": 0.04, "costume": 0.05 },
  "ocr":                 { "scoreThreshold": 0.5, "detScales": [960, 1600] },
  "wdTagger":            { "characterTagThreshold": 0.6 },
  "characterAliases":    { "uehara_ayumu": "上原歩夢" }
}
```

Thresholds are **not comparable across tasks** — each model has its own cosine distribution. The face
value of 0.28 is empirical: same person scores 0.64–0.79, a photographed photo of them 0.32, a
different person 0.15. A value like 0.8 rejects even genuine matches.

A match must clear both the absolute threshold and the **top-1 minus top-2 margin**. If top-1 and
top-2 are close, the match is meaningless regardless of its absolute score, and reporting "unknown"
beats reporting a wrong name.

`characterAliases` maps WD-Tagger's romanized danbooru tags to whatever naming your gallery uses.

### Search lists

One line per list. Tab-separated parts within a line must **all** be present (AND); lines are OR'd,
so **any one list matching** counts as a hit.

```
大西亜玖璃
上原歩夢	虹ヶ咲
```

```bash
npm run search-strings -- --add "大西亜玖璃"
npm run search-strings -- --list
```

## Diagnostics

```bash
npm run analyze -- sample/           # analyze a file or folder without the server
npm run tags -- sample/image.jpg     # dump WD-Tagger raw probabilities
npm run detect -- sample/image.jpg   # dump detection boxes and face landmarks
```

`npm run analyze` uses the same code path as the server, so it is the tool for threshold tuning and
regression checks.

## Performance

Roughly 7–40 s per image on an i5-3550 (no AVX2). Recognition dominates: the multilingual OCR model
is 84 MB with 18,385 output classes, costing 0.8–1.3 s per text strip, so text-heavy images are the
slow case. Line merging, minimum-strip-width filtering, scale de-duplication, batched inference, and
a region cap keep the worst case bounded. INT8 quantization of the recognition model is the main
remaining lever.

The lightweight PP-OCRv4 mobile recognizer cannot be substituted: its dictionary contains only 5 kana
and lacks `ヶ` and `咲` entirely, making Japanese impossible.

## Resetting

Delete `db/embeddings.db` and re-run `npm run register:all`.

## Models

All auto-downloaded to `models/` on first run.

| File | Model | Purpose |
|---|---|---|
| `ocr-det.onnx` | PP-OCRv4 DBNet | text region detection |
| `ocr-rec-ch.onnx` + dict | PaddleOCR multilingual CRNN | CJK / kana / latin recognition |
| `ocr-rec-ko.onnx` + dict | PaddleOCR Korean CRNN | hangul recognition |
| `face-det.onnx` | InsightFace SCRFD det_10g | real face detection + 5 landmarks |
| `arcface-w600k-r50.onnx` | InsightFace ArcFace | face embedding |
| `anime-face-det.onnx` | deepghs YOLOv8s | anime face detection |
| `anime-person-det.onnx` | deepghs YOLOv8s | anime person detection |
| `ccip-feat.onnx` | deepghs CCIP caformer | anime character identity embedding |
| `wd-tagger.onnx` + tags | SmilingWolf wd-vit-tagger-v3 | character names + clothing tags |
