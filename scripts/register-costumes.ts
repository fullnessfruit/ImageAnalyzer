/**
 * Register costume embeddings from data/costumes/ into SQLite DB.
 *
 * Each subfolder name = character name whose costumes they are.
 * Extracts CLIP embeddings and stores in db/embeddings.db.
 * Already registered images are skipped.
 *
 * Usage: npx ts-node scripts/register-costumes.ts
 */

import path from "path";
import fs from "fs";
import { initDB, isImageRegistered, insertEmbedding } from "../server/src/db";
import { ensureModelsDownloaded } from "../server/src/model-downloader";
import { loadModels, extractCLIPEmbedding } from "../server/src/inference";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const COSTUMES_DIR = path.join(PROJECT_ROOT, "data", "costumes");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");
const DB_DIR = path.join(PROJECT_ROOT, "db");

const SUPPORTED_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

async function main() {
  console.log("Registering costume embeddings...");

  await ensureModelsDownloaded(MODELS_DIR);
  initDB(DB_DIR);
  await loadModels(MODELS_DIR);

  if (!fs.existsSync(COSTUMES_DIR)) {
    console.log(`Costumes directory not found: ${COSTUMES_DIR}`);
    return;
  }

  const folders = fs.readdirSync(COSTUMES_DIR).filter((f) =>
    fs.statSync(path.join(COSTUMES_DIR, f)).isDirectory()
  );

  if (folders.length === 0) {
    console.log("No costume folders found.");
    return;
  }

  let registered = 0;
  let skipped = 0;

  for (const name of folders) {
    const folderPath = path.join(COSTUMES_DIR, name);
    const images = fs.readdirSync(folderPath).filter((f) =>
      SUPPORTED_EXT.has(path.extname(f).toLowerCase())
    );

    for (const imgFile of images) {
      const imgPath = path.join(folderPath, imgFile);
      const relPath = path.relative(PROJECT_ROOT, imgPath);

      if (isImageRegistered("costume_embeddings", relPath)) {
        skipped++;
        continue;
      }

      const buffer = fs.readFileSync(imgPath);
      const embedding = await extractCLIPEmbedding(buffer);
      if (!embedding) {
        console.error(`Failed to extract embedding for ${relPath}`);
        continue;
      }

      insertEmbedding("costume_embeddings", name, relPath, embedding);
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
