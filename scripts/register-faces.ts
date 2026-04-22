/**
 * Register face embeddings from data/faces/ into SQLite DB.
 *
 * Each subfolder name = person name.
 * Detects faces, extracts ArcFace embeddings, and stores in db/embeddings.db.
 * Already registered images are skipped.
 *
 * Usage: npx ts-node scripts/register-faces.ts
 */

import path from "path";
import fs from "fs";
import { initDB, isImageRegistered, insertEmbedding } from "../server/src/db";
import { ensureModelsDownloaded } from "../server/src/model-downloader";
import { loadModels, detectFaces, extractArcFaceEmbedding, cropRegion } from "../server/src/inference";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(PROJECT_ROOT, "data");
const FACES_DIR = path.join(DATA_DIR, "faces");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");
const DB_DIR = path.join(PROJECT_ROOT, "db");

const SUPPORTED_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

async function main() {
  console.log("Registering face embeddings...");

  await ensureModelsDownloaded(MODELS_DIR);
  initDB(DB_DIR);
  await loadModels(MODELS_DIR);

  if (!fs.existsSync(FACES_DIR)) {
    console.log(`Faces directory not found: ${FACES_DIR}`);
    return;
  }

  const personFolders = fs.readdirSync(FACES_DIR).filter((f) =>
    fs.statSync(path.join(FACES_DIR, f)).isDirectory()
  );

  if (personFolders.length === 0) {
    console.log("No person folders found.");
    return;
  }

  let registered = 0;
  let skipped = 0;

  for (const personName of personFolders) {
    const personPath = path.join(FACES_DIR, personName);
    const images = fs.readdirSync(personPath).filter((f) =>
      SUPPORTED_EXT.has(path.extname(f).toLowerCase())
    );

    for (const imgFile of images) {
      const imgPath = path.join(personPath, imgFile);
      const relPath = path.relative(PROJECT_ROOT, imgPath);

      if (isImageRegistered("face_embeddings", relPath)) {
        skipped++;
        continue;
      }

      const buffer = fs.readFileSync(imgPath);
      const faces = await detectFaces(buffer);

      if (faces.length === 0) {
        console.warn(`No face detected in ${relPath}, skipping.`);
        continue;
      }

      // Use the largest detected face
      const bestFace = faces.reduce((a, b) =>
        a.box[2] * a.box[3] > b.box[2] * b.box[3] ? a : b
      );

      const cropped = await cropRegion(buffer, bestFace.box);
      const embedding = await extractArcFaceEmbedding(cropped);
      if (!embedding) {
        console.error(`Failed to extract embedding for ${relPath}`);
        continue;
      }

      insertEmbedding("face_embeddings", personName, relPath, embedding);
      registered++;
      console.log(`Registered: ${personName} — ${imgFile}`);
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
