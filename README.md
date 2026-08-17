# ImageAnalyzer

Local API server for image analysis. Accepts an image and answers four questions, designed to be
called from a Chrome extension:

1. **OCR list matching** - does the image contain any of the strings in `searchStrings.tsv`?
2. **Real-person face** - is a registered person's face present?
3. **Anime character** - which character is this?
4. **Anime costume** - is a registered costume being worn?

It does not extract full text, describe scenes, identify silhouettes, or do real-person full-body
re-identification. Those are out of scope by design.

## How it works

**OCR is treated as keyword spotting, not reading.** The search list is known in advance, so instead
of decoding text and comparing substrings, each candidate string is scored directly against the CTC
probability lattice. A greedy decode picks the top class at every timestep and its exact-match
probability decays as the L-th power of per-character accuracy; lattice scoring survives a character
being pushed to second place. Observed: a region whose greedy decode substituted a visually
near-identical character still scored 0.83 for the intended string, and a second target went from
0.22 to 0.93 once that confusion pair was added to the per-position alternatives.

**Identity matching uses task-appropriate embedding spaces.** CLIP is not used anywhere - its space
clusters by art style and composition rather than identity, so the same character in a different style
lands far away while different characters with similar attributes land close.

| Task | Model | Registration needed |
|---|---|---|
| Real-person face | SCRFD landmarks → 5-point similarity warp → ArcFace w600k_r50 (flip TTA) | 1–3 photos per person |
| Anime character | CCIP (gallery), corroborated by WD-Tagger v3 | 1–2 images per character |
| Anime costume | WD-Tagger clothing tags, IDF-weighted, head masked | 1 image per costume |

**Only registered characters are reported.** CCIP carries identity: it is trained with the same
character across different artists and styles as positives, so a single registered image matches the
character in a different outfit and a different rendering (measured: 0.85 and 0.87 from one reference).
WD-Tagger is a corroborating signal only - its vocabulary is a fixed ~2,751 characters and it does not
know newer ones at all (a recent franchise's entire twelve-character cast can be absent from it), so
any name it emits is filtered against the gallery.

ArcFace is trained only on faces warped to a canonical 5-point template, so landmark alignment is
mandatory - feeding a raw bounding-box crop puts every input out of distribution.

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

Folder name = the name reported by the analyzer. Faces do not need cropping - detection handles it.

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
    "found": ["search-string-a"],
    "detail": [{ "list": "search-string-a", "matched": true, "parts": [{ "text": "search-string-a", "score": 0.834 }] }],
    "regions": 19
  },
  "faces":      [{ "name": "person-a", "score": 0.7854, "margin": 0.7854, "box": [x, y, w, h] }],
  "facesWeak":  [{ "name": "person-a", "score": 0.3221, "margin": 0.3221, "box": [x, y, w, h] }],
  "characters": [{ "name": "character-a", "score": 0.8731, "margin": 0.8731, "source": "ccip" }],
  "costumes":   [{ "name": "costume-a", "score": 1.0, "margin": 1.0, "source": "blazer,skirt,..." }],
  "_detections": { "realFaces": 2, "animeFaces": 2, "animePersons": 2, "textRegions": 19 },
  "_elapsedMs": 30607
}
```

`GET /health` reports which models loaded and how many gallery entries exist per task.

## Configuration

`config.json` and `searchStrings.tsv` are re-read on every request - no restart needed.

```json
{
  "similarityThreshold": { "face": 0.45, "faceWeak": 0.28, "character": 0.82, "costume": 0.55 },
  "margin":              { "face": 0.06, "character": 0.04, "costume": 0.05 },
  "ocr":                 { "scoreThreshold": 0.5, "detScales": [960, 1600] },
  "wdTagger":            { "characterTagThreshold": 0.6 },
  "characterAliases":    { "tagger_tag_name": "character-a" },
  "candidates":          { "enabled": true, "dedupThreshold": 0.95, "maxPerName": 50 }
}
```

Thresholds are **not comparable across tasks** - each model has its own cosine distribution. A value
like 0.8, reasonable for CCIP, rejects every genuine ArcFace match.

Faces are reported in two bands so that a photographed photo is not accepted as the real thing but is
not silently dropped either. Measured on one image containing both: the live subject scores 0.7854 and
lands in `faces`; the framed photo of her scores 0.3221 and lands in `facesWeak`; a different person
scores 0.1474 and is not reported at all.

| Band | Range | Meaning |
|---|---|---|
| `faces` | ≥ `face` (0.45) | accepted as the real subject |
| `facesWeak` | `faceWeak` … `face` | photographed photo or degraded capture - do not act on automatically |
| - | < `faceWeak` (0.28) | not reported |

A match must clear both the absolute threshold and the **top-1 minus top-2 margin**. If top-1 and
top-2 are close, the match is meaningless regardless of its absolute score, and reporting "unknown"
beats reporting a wrong name.

`characterAliases` maps WD-Tagger's tag names to whatever naming your gallery uses.

### Search lists

One line per list. Tab-separated parts within a line must **all** be present (AND); lines are OR'd,
so **any one list matching** counts as a hit.

```
search-string-a
first-part	second-part
```

```bash
npm run search-strings -- --add "search-string-a"
npm run search-strings -- --list
```

## Growing the gallery

Nothing is ever added to the gallery automatically. When a crop matches a registered identity with
confidence, the crop is written to `data/_candidates/<kind>/<name>/` and recorded separately - the
gallery itself is untouched. You review the folder and promote what is worth keeping.

```bash
npm run candidates -- --list                      # what has been collected
npm run candidates -- --promote face person-a     # move into data/faces/person-a/
npm run register:faces                            # actually register
npm run candidates -- --clear                     # discard everything collected
```

Auto-registration was rejected deliberately: a wrong entry silently changes later matching and feeds
the next wrong entry, and since the crop exists only in memory there would be nothing to inspect -
the only possible undo is deleting every auto entry blindly. Reviewing costs almost nothing here
because there is no training; registering is embedding extraction into SQLite.

Unmatched crops are not collected. The purpose is to broaden coverage of identities you already
registered, and for faces only the `faces` band is collected - a photographed photo must not end up
in the gallery. For characters the most valuable candidate is one WD-Tagger confirms but CCIP misses,
since that is exactly a rendering CCIP does not yet cover.

Runaway collection is bounded by `dedupThreshold` (skip anything within 0.95 cosine of an existing
candidate - the embedding is already computed, so this is free) and `maxPerName`.

## Diagnostics

```bash
npm run analyze -- sample/           # analyze a file or folder without the server
npm run tags -- sample/image.jpg     # dump WD-Tagger raw probabilities
npm run detect -- sample/image.jpg   # dump detection boxes and face landmarks
```

`npm run analyze` uses the same code path as the server, so it is the tool for threshold tuning and
regression checks.

## Resetting

Delete `db/embeddings.db` and re-run `npm run register:all`.
