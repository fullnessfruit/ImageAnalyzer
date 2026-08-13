/**
 * 로컬 이미지를 서버 없이 분석한다. 서버와 동일한 analyzeImage 경로를 쓰므로
 * 임계값 조정과 회귀 확인에 그대로 쓸 수 있다.
 *
 * Usage:
 *   npx ts-node scripts/analyze.ts <이미지경로 또는 폴더> [--json]
 */

import path from "path";
import fs from "fs";
import { initDB } from "../server/src/db";
import { ensureModelsDownloaded } from "../server/src/model-downloader";
import { initOCR, parseSearchLists } from "../server/src/ocr";
import { loadModels } from "../server/src/inference";
import { analyzeImage } from "../server/src/analyze";
import { loadConfig } from "../server/src/config";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SUPPORTED_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

async function main() {
  const target = process.argv[2];
  const asJson = process.argv.includes("--json");
  if (!target) {
    console.error("Usage: npx ts-node scripts/analyze.ts <이미지경로 또는 폴더> [--json]");
    process.exit(1);
  }

  const abs = path.resolve(PROJECT_ROOT, target);
  const files = fs.statSync(abs).isDirectory()
    ? fs.readdirSync(abs).filter((f) => SUPPORTED_EXT.has(path.extname(f).toLowerCase())).map((f) => path.join(abs, f))
    : [abs];

  await ensureModelsDownloaded(path.join(PROJECT_ROOT, "models"));
  initDB(path.join(PROJECT_ROOT, "db"));
  await loadModels(path.join(PROJECT_ROOT, "models"));
  await initOCR(path.join(PROJECT_ROOT, "models"));

  const ssPath = path.join(PROJECT_ROOT, "searchStrings.tsv");
  const searchLists = fs.existsSync(ssPath) ? parseSearchLists(fs.readFileSync(ssPath, "utf-8")) : [];
  const cfg = loadConfig();

  for (const file of files) {
    console.log(`\n${"=".repeat(70)}\n${path.basename(file)}\n${"=".repeat(70)}`);
    const result = await analyzeImage(fs.readFileSync(file), searchLists, cfg);

    if (asJson) {
      console.log(JSON.stringify(result, null, 2));
      continue;
    }

    console.log(`  detections   : ${JSON.stringify(result._detections)}`);
    console.log(`  ocr.found    : ${JSON.stringify(result.ocr.found)}`);
    console.log(`  ocr.text     : ${result.ocr.fullText.slice(0, 200)}`);
    console.log(`  faces        : ${result.faces.map((m) => `${m.name}(${m.score})`).join(", ") || "-"}`);
    console.log(`  facesWeak    : ${result.facesWeak.map((m) => `${m.name}(${m.score})`).join(", ") || "-"}`);
    console.log(`  characters   : ${result.characters.map((m) => `${m.name}[${m.source}](${m.score})`).join(", ") || "-"}`);
    console.log(`  costumes     : ${result.costumes.map((m) => `${m.name}(${m.score})`).join(", ") || "-"}`);
    console.log(`  elapsed      : ${result._elapsedMs}ms`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
