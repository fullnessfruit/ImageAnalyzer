import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";

import { initDB, getAllEmbeddings, EmbeddingRow } from "./db";
import { ensureModelsDownloaded } from "./model-downloader";
import { initOCR, performOCR } from "./ocr";
import {
  loadModels,
  detectWithYOLO,
  detectFaces,
  detectPersons,
  extractCLIPEmbedding,
  extractArcFaceEmbedding,
  extractCostumeRegion,
  cosineSimilarity,
  cropRegion,
  Detection,
} from "./inference";

const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");
const DB_DIR = path.join(PROJECT_ROOT, "db");
const CONFIG_PATH = path.join(PROJECT_ROOT, "config.json");
const SEARCH_STRINGS_PATH = path.join(PROJECT_ROOT, "searchStrings.tsv");

interface Config {
  similarityThreshold: {
    character: number;
    face: number;
    costume: number;
    person: number;
  };
}

function loadConfig(): Config {
  const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
  return JSON.parse(raw) as Config;
}

function loadSearchStrings(): string[] {
  if (!fs.existsSync(SEARCH_STRINGS_PATH)) return [];
  return fs
    .readFileSync(SEARCH_STRINGS_PATH, "utf-8")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0);
}

// --- Shared matching helper ---

interface RecognitionMatch {
  name: string;
  confidence: number;
  box: [number, number, number, number];
}

function findBestMatch(embedding: Float32Array, dbEmbeddings: EmbeddingRow[], threshold: number, label?: string): { name: string; similarity: number } | null {
  let best = { name: "", similarity: 0 };
  for (const row of dbEmbeddings) {
    const stored = new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4);
    const sim = cosineSimilarity(embedding, stored);
    if (sim > best.similarity) {
      best = { name: row.name, similarity: sim };
    }
  }
  if (label) {
    const status = best.similarity >= threshold ? "MATCH" : "BELOW";
    console.log(`[${label}] best="${best.name}" sim=${best.similarity.toFixed(4)} thr=${threshold} → ${status}`);
  }
  return best.similarity >= threshold ? best : null;
}

// --- Character Recognition ---

async function recognizeCharacters(imageBuffer: Buffer, detections: Detection[], threshold: number): Promise<RecognitionMatch[]> {
  const characterDetections = detections.filter((d) => d.classId === 0);
  if (characterDetections.length === 0) return [];

  const dbEmbeddings = getAllEmbeddings("character_embeddings");
  if (dbEmbeddings.length === 0) return [];

  const matches: RecognitionMatch[] = [];
  for (const det of characterDetections) {
    const cropped = await cropRegion(imageBuffer, det.box);
    const embedding = await extractCLIPEmbedding(cropped);
    if (!embedding) continue;

    const best = findBestMatch(embedding, dbEmbeddings, threshold, "character");
    if (best) {
      matches.push({ name: best.name, confidence: parseFloat(best.similarity.toFixed(4)), box: det.box });
    }
  }
  return matches;
}

// --- Face Recognition ---

async function recognizeFaces(imageBuffer: Buffer, detections: Detection[], threshold: number): Promise<RecognitionMatch[]> {
  if (detections.length === 0) return [];

  const dbEmbeddings = getAllEmbeddings("face_embeddings");
  if (dbEmbeddings.length === 0) return [];

  const matches: RecognitionMatch[] = [];
  for (const det of detections) {
    const cropped = await cropRegion(imageBuffer, det.box);
    const embedding = await extractArcFaceEmbedding(cropped);
    if (!embedding) continue;

    const best = findBestMatch(embedding, dbEmbeddings, threshold, "face");
    if (best) {
      matches.push({ name: best.name, confidence: parseFloat(best.similarity.toFixed(4)), box: det.box });
    }
  }
  return matches;
}

// --- Costume Recognition ---

async function recognizeCostumes(
  imageBuffer: Buffer,
  characterDetections: Detection[],
  faceDetections: Detection[],
  threshold: number
): Promise<RecognitionMatch[]> {
  const charDets = characterDetections.filter((d) => d.classId === 0);
  if (charDets.length === 0) return [];

  const dbEmbeddings = getAllEmbeddings("costume_embeddings");
  if (dbEmbeddings.length === 0) return [];

  const matches: RecognitionMatch[] = [];
  for (const det of charDets) {
    const costumeRegion = await extractCostumeRegion(imageBuffer, det.box, faceDetections);
    if (!costumeRegion) continue;

    const embedding = await extractCLIPEmbedding(costumeRegion);
    if (!embedding) continue;

    const best = findBestMatch(embedding, dbEmbeddings, threshold, "costume");
    if (best) {
      matches.push({ name: best.name, confidence: parseFloat(best.similarity.toFixed(4)), box: det.box });
    }
  }
  return matches;
}

// --- Person Recognition (full-body ReID) ---

async function recognizePersons(
  imageBuffer: Buffer,
  personDetections: Detection[],
  faceDetections: Detection[],
  threshold: number
): Promise<RecognitionMatch[]> {
  if (personDetections.length === 0) return [];

  const dbEmbeddings = getAllEmbeddings("person_embeddings");
  if (dbEmbeddings.length === 0) return [];

  const matches: RecognitionMatch[] = [];
  for (const det of personDetections) {
    const masked = await extractCostumeRegion(imageBuffer, det.box, faceDetections);
    // Fallback: if no face found in person box, use raw crop (same as register-persons.ts)
    const region = masked ?? await cropRegion(imageBuffer, det.box);

    const embedding = await extractCLIPEmbedding(region);
    if (!embedding) continue;

    const best = findBestMatch(embedding, dbEmbeddings, threshold, "person");
    if (best) {
      matches.push({ name: best.name, confidence: parseFloat(best.similarity.toFixed(4)), box: det.box });
    }
  }
  return matches;
}

// --- Server ---

async function main() {
  console.log("Initializing ImageAnalyzer server...");

  await ensureModelsDownloaded(MODELS_DIR);
  initDB(DB_DIR);
  await loadModels(MODELS_DIR);
  await initOCR(MODELS_DIR);

  const app = express();
  app.use((_, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    res.header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    next();
  });

  const upload = multer({ storage: multer.memoryStorage() });

  app.post("/analyze", upload.single("image"), async (req, res) => {
    if (!req.file) {
      res.status(400).json({ error: "No image file provided. Use field name 'image'." });
      return;
    }

    const config = loadConfig();
    const searchStrings = loadSearchStrings();
    const imageBuffer = req.file.buffer;

    try {
      const [ocr, yoloDetections, faceDetections, personDetections] = await Promise.all([
        performOCR(imageBuffer, searchStrings),
        detectWithYOLO(imageBuffer),
        detectFaces(imageBuffer),
        detectPersons(imageBuffer),
      ]);

      const [characters, faces, costumes, persons] = await Promise.all([
        recognizeCharacters(imageBuffer, yoloDetections, config.similarityThreshold.character),
        recognizeFaces(imageBuffer, faceDetections, config.similarityThreshold.face),
        recognizeCostumes(imageBuffer, yoloDetections, faceDetections, config.similarityThreshold.costume),
        recognizePersons(imageBuffer, personDetections, faceDetections, config.similarityThreshold.person),
      ]);

      res.json({
        ocr,
        characters,
        faces,
        costumes,
        persons,
        _detections: {
          yoloCharacters: yoloDetections.filter((d) => d.classId === 0).length,
          yoloFaces: yoloDetections.filter((d) => d.classId === 1).length,
          faceDetFaces: faceDetections.length,
          persons: personDetections.length,
        },
      });
    } catch (err: any) {
      console.error(`Analysis error: ${err.message}`);
      res.status(500).json({ error: "Analysis failed", details: err.message });
    }
  });

  app.get("/health", (_, res) => {
    res.json({ status: "ok" });
  });

  const PORT = 3000;
  app.listen(PORT, () => {
    console.log(`ImageAnalyzer server running on http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error("Server startup failed:", err);
  process.exit(1);
});
