# ImageAnalyzer

Local API server for image analysis. Accepts an image and performs five analyses: **OCR**, **Character Recognition**, **Face Recognition**, **Costume Recognition**, and **Person Recognition (full-body ReID)**. Designed to be called from a Chrome extension.

## Architecture

- **Training** (Python): YOLOv8 fine-tuning for character/face region detection
- **Server** (Node.js + TypeScript): Analysis API using ONNX models
- **Scripts** (TypeScript): Embedding registration utilities

## Folder Structure

```
ImageAnalyzer/
├── data/
│   ├── characters/          # Character subfolders (e.g., pikachu/, naruto/)
│   ├── faces/               # Person subfolders for face recognition
│   ├── costumes/            # Costume subfolders by character name
│   ├── persons/             # Person subfolders for full-body recognition
│   └── yolo-training/       # YOLO training images + labels
│       ├── images/
│       └── labels/
├── models/                  # ONNX model storage
├── db/                      # SQLite database
├── training/                # Python training scripts
├── server/                  # Node.js analysis server
├── scripts/                 # Utility scripts (registration, config)
├── config.json              # Similarity thresholds
├── searchStrings.tsv        # OCR search strings (one per line)
├── setup.sh                 # Initial setup script (Linux/macOS)
├── setup.bat                # Initial setup script (Windows)
└── package.json
```

## Recognition Pipeline

Detection and identification models have separate roles:

1. **YOLO (custom)** — Detects character/face **regions** in the image (does not identify who)
2. **YOLO (pretrained COCO)** — Detects real **person regions** in the image
3. **CLIP** — Extracts embeddings from cropped regions → cosine similarity against DB → **identifies character, costume, or person (full-body)**
4. **ArcFace** — Extracts embeddings from cropped face regions → cosine similarity against DB → **identifies person by face**

## Data Preparation

### data/characters/

**Used for**: YOLO training data + CLIP embedding registration (dual purpose)

```
data/characters/
  ├─ character-a/
  │   ├─ illust1.png
  │   └─ sketch1.png
  └─ character-b/
      └─ illust1.png
```

- **Folder name = character name** (used as label when registering CLIP embeddings)
- **Images must be tightly cropped to a single character** — YOLO training uses the entire image as a bounding box, so multiple characters in one image will cause incorrect training
- **Mixed art styles OK** — Illustrations, doodles, sketches in the same folder help YOLO learn to detect various styles, and provide broader CLIP embedding coverage

### data/faces/

**Used for**: ArcFace embedding registration only (not used for YOLO training)

```
data/faces/
  ├─ person-a/
  │   └─ photo1.jpg
  └─ person-b/
      └─ photo1.jpg
```

- **Folder name = person name**
- **Tight cropping not required** — The face detection model automatically finds and crops the face region. Any photo containing a face will work

### data/costumes/

**Used for**: Costume CLIP embedding registration only (not used for YOLO training)

```
data/costumes/
  ├─ character-a/    ← Costumes previously worn by character-a
  │   ├─ costume1.png
  │   └─ costume2.png
  └─ character-b/
      └─ costume1.png
```

- **Folder name = name of the character who wore these costumes**
- Multiple costumes in one folder are matched as "a costume that character-a has worn"
- **Crop to show only the costume (exclude face)** — During analysis, faces are auto-masked. Registration images without faces improve matching accuracy

### data/persons/

**Used for**: Full-body CLIP embedding registration (not used for YOLO training)

```
data/persons/
  ├─ person-a/
  │   └─ fullbody1.jpg
  └─ person-b/
      └─ fullbody1.jpg
```

- **Folder name = person name**
- Use full-body photos showing the person's typical outfit
- Faces are automatically detected and masked during registration, so the embedding captures body/clothing features only
- **Clothing-dependent**: same person in different outfits will not match. Register multiple photos per outfit for better accuracy

## Quick Start

### 1. Initial Setup

```bash
# Linux / macOS
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

This will install dependencies, download required ONNX models (CLIP, ArcFace, face detection, person detection), and initialize the database.

If the person detection model (yolo-person.onnx) auto-download fails:
```bash
cd training
python export_person_model.py
```

### 2. Training (Optional)

If you want to train a custom YOLO model for character/face detection:

```bash
cd training
pip install -r requirements.txt

# Place character images in data/characters/<character-name>/
python augment.py    # Generate augmented training data
python train.py      # Train YOLOv8 and export to ONNX
```

To add new characters: add images to `data/characters/` and re-run augment.py → train.py. Each run retrains from scratch using all data, so there is no need to delete existing images.

For Google Colab:
```bash
python train.py --data-dir /content/drive/MyDrive/ImageAnalyzer/data --models-dir /content/drive/MyDrive/ImageAnalyzer/models
```

### 3. Register Embeddings

Place reference images in the appropriate directories:
- `data/characters/<character-name>/` for character images
- `data/faces/<person-name>/` for face photos
- `data/costumes/<character-name>/` for costume images
- `data/persons/<person-name>/` for full-body photos

```bash
npm run register:characters  # Register character embeddings
npm run register:faces       # Register face embeddings
npm run register:costumes    # Register costume embeddings
npm run register:persons     # Register person (full-body) embeddings
npm run register:all         # Register all
```

### 4. Start Server

```bash
npm run server   # Production
npm run dev      # Development (auto-reload)
```

Server runs on `http://localhost:3000`.

### 5. API Usage

**POST /analyze**

Send an image as multipart/form-data with field name `image`:

```bash
curl -X POST http://localhost:3000/analyze -F "image=@test.jpg"
```

**Response:**

```json
{
  "ocr": {
    "found": ["matched-string"],
    "fullText": "full extracted text"
  },
  "characters": [
    { "name": "pikachu", "confidence": 0.92, "box": [100, 50, 200, 300] }
  ],
  "faces": [
    { "name": "person-a", "confidence": 0.95, "box": [150, 30, 120, 150] }
  ],
  "costumes": [
    { "name": "pikachu", "confidence": 0.85, "box": [100, 50, 200, 300] }
  ],
  "persons": [
    { "name": "person-a", "confidence": 0.78, "box": [80, 20, 250, 500] }
  ]
}
```

## Configuration

### Similarity Thresholds

Edit `config.json`:

```json
{
  "similarityThreshold": {
    "character": 0.75,
    "face": 0.80,
    "costume": 0.75,
    "person": 0.70
  }
}
```

### OCR Search Strings

Edit `searchStrings.tsv` (one entry per line, re-read on every request):

```
keyword
#Name	Uehara Ayumu	大西亜玖璃
```

- **Simple match** — a line without tabs matches if found anywhere in the extracted text
- **Multi-part match** — a line with tabs requires ALL tab-separated parts to be present independently in the image (they do not need to be adjacent)

Or use the CLI tool:

```bash
npx ts-node scripts/update-config.ts --add "new string"
npx ts-node scripts/update-config.ts --remove "old string"
npx ts-node scripts/update-config.ts --list
```

## Resetting Trained Data

- **Reset embeddings**: Delete `db/embeddings.db`, then re-run `npm run register:all`
- **Reset YOLO model**: Delete `models/yolo-characters.onnx` and `models/runs/`, then re-run `augment.py` → `train.py`

## Models

| Model | Purpose | Auto-downloaded |
|-------|---------|-----------------|
| CLIP ViT-B/32 | Character/costume/person embedding extraction | Yes |
| ArcFace w600k_r50 | Face embedding extraction | Yes |
| det_10g | Face detection | Yes |
| YOLOv8n (COCO) | Person region detection | Yes (fallback: `python training/export_person_model.py`) |
| yolo-characters | Character/face region detection | No (trained locally) |

## Health Check

```bash
curl http://localhost:3000/health
```
