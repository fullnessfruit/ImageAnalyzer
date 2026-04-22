/**
 * Register person embeddings from data/persons/ into SQLite DB.
 *
 * Each subfolder name = person name.
 * Detects and masks faces, then extracts CLIP embeddings for full-body ReID.
 * Already registered images are skipped.
 *
 * Usage: npx ts-node scripts/register-persons.ts
 */

import path from "path";
import fs from "fs";
import sharp from "sharp";
import { initDB, isImageRegistered, insertEmbedding } from "../server/src/db";
import { ensureModelsDownloaded } from "../server/src/model-downloader";
import {
  loadModels,
  detectFaces,
  extractCostumeRegion,
  extractCLIPEmbedding,
  Detection,
} from "../server/src/inference";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const PERSONS_DIR = path.join(PROJECT_ROOT, "data", "persons");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");
const DB_DIR = path.join(PROJECT_ROOT, "db");

const SUPPORTED_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

async function main() {
  console.log("Registering person embeddings...");

  await ensureModelsDownloaded(MODELS_DIR);
  initDB(DB_DIR);
  await loadModels(MODELS_DIR);

  if (!fs.existsSync(PERSONS_DIR)) {
    console.log(`Persons directory not found: ${PERSONS_DIR}`);
    return;
  }

  const folders = fs.readdirSync(PERSONS_DIR).filter((f) =>
    fs.statSync(path.join(PERSONS_DIR, f)).isDirectory()
  );

  if (folders.length === 0) {
    console.log("No person folders found.");
    return;
  }

  let registered = 0;
  let skipped = 0;

  for (const name of folders) {
    const folderPath = path.join(PERSONS_DIR, name);
    const images = fs.readdirSync(folderPath).filter((f) =>
      SUPPORTED_EXT.has(path.extname(f).toLowerCase())
    );

    for (const imgFile of images) {
      const imgPath = path.join(folderPath, imgFile);
      const relPath = path.relative(PROJECT_ROOT, imgPath);

      if (isImageRegistered("person_embeddings", relPath)) {
        skipped++;
        continue;
      }

      const buffer = fs.readFileSync(imgPath);

      // Mask face so embedding captures body/clothing only
      const faces = await detectFaces(buffer);
      let processedBuffer: Buffer = buffer;
      if (faces.length > 0) {
        const meta = await sharp(buffer).metadata();
        const fullBox: [number, number, number, number] = [0, 0, meta.width!, meta.height!];
        const masked = await extractCostumeRegion(buffer, fullBox, faces);
        if (masked) processedBuffer = Buffer.from(masked);
      }

      const embedding = await extractCLIPEmbedding(processedBuffer);
      if (!embedding) {
        console.error(`Failed to extract embedding for ${relPath}`);
        continue;
      }

      insertEmbedding("person_embeddings", name, relPath, embedding);
      registered++;
      console.log(`Registered: ${name} — ${imgFile}`);
    }
  }

  console.log(`Done - registered: ${registered}, skipped: ${skipped}`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("Registration failed:", err);
    process.exit(1);
  });
