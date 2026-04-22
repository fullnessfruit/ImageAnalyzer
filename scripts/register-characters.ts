/**
 * Register character embeddings from data/characters/ into SQLite DB.
 *
 * Each subfolder name = character name.
 * Extracts CLIP embeddings and stores in db/embeddings.db.
 * Already registered images are skipped.
 *
 * Usage: npx ts-node scripts/register-characters.ts
 */

import path from "path";
import fs from "fs";
import { initDB, isImageRegistered, insertEmbedding } from "../server/src/db";
import { ensureModelsDownloaded } from "../server/src/model-downloader";
import { loadModels, extractCLIPEmbedding } from "../server/src/inference";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(PROJECT_ROOT, "data");
const CHARACTERS_DIR = path.join(DATA_DIR, "characters");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");
const DB_DIR = path.join(PROJECT_ROOT, "db");

const SUPPORTED_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

async function main() {
  console.log("Registering character embeddings...");

  await ensureModelsDownloaded(MODELS_DIR);
  initDB(DB_DIR);
  await loadModels(MODELS_DIR);

  if (!fs.existsSync(CHARACTERS_DIR)) {
    console.log(`Characters directory not found: ${CHARACTERS_DIR}`);
    return;
  }

  const charFolders = fs.readdirSync(CHARACTERS_DIR).filter((f) =>
    fs.statSync(path.join(CHARACTERS_DIR, f)).isDirectory()
  );

  if (charFolders.length === 0) {
    console.log("No character folders found.");
    return;
  }

  let registered = 0;
  let skipped = 0;

  for (const charName of charFolders) {
    const charPath = path.join(CHARACTERS_DIR, charName);
    const images = fs.readdirSync(charPath).filter((f) =>
      SUPPORTED_EXT.has(path.extname(f).toLowerCase())
    );

    for (const imgFile of images) {
      const imgPath = path.join(charPath, imgFile);
      const relPath = path.relative(PROJECT_ROOT, imgPath);

      if (isImageRegistered("character_embeddings", relPath)) {
        skipped++;
        continue;
      }

      const buffer = fs.readFileSync(imgPath);
      const embedding = await extractCLIPEmbedding(buffer);
      if (!embedding) {
        console.error(`Failed to extract embedding for ${relPath}`);
        continue;
      }

      insertEmbedding("character_embeddings", charName, relPath, embedding);
      registered++;
      console.log(`Registered: ${charName} — ${imgFile}`);
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
