/**
 * 로컬 이미지를 서버 없이 분석한다. 서버와 동일한 analyzeImage 경로를 쓰므로
 * 임계값 조정과 회귀 확인에 그대로 쓸 수 있다.
 *
 * Usage:
 *   npx ts-node scripts/analyze.ts <이미지경로 또는 폴더> [--json]
 */

import path from "path";
import fs from "fs";
import nodeCrypto from "crypto";
import { pathToFileURL } from "url";
import { initDB } from "../server/src/db";
import { ensureModelsDownloaded } from "../server/src/model-downloader";
import { initOCR, parseSearchLists } from "../server/src/ocr";
import { loadModels } from "../server/src/inference";
import { analyzeImage } from "../server/src/analyze";
import { loadConfig } from "../server/src/config";
import { appendAnalysisLog } from "../server/src/analysis-log";

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
    const image = fs.readFileSync(file);
    const imageRef = {
      mode: "cli" as const,
      requestId: nodeCrypto.randomUUID(),
      imageUrl: pathToFileURL(file).href,
      imageName: path.basename(file),
      imageSha256: nodeCrypto.createHash("sha256").update(image).digest("hex"),
    };
    appendAnalysisLog({
      event: "image_request", ...imageRef, listSource: "file",
      searchStrings: searchLists.map((list) => list.raw).join("\n"), config: cfg,
    });
    let result;
    try {
      result = await analyzeImage(image, searchLists, cfg);
    } catch (err: any) {
      appendAnalysisLog({ event: "image_result", ...imageRef, status: "analysis_failed", error: err.message });
      throw err;
    }
    appendAnalysisLog({ event: "image_result", ...imageRef, status: "analyzed", result });

    if (asJson) {
      console.log(JSON.stringify(result, null, 2));
      continue;
    }

    console.log(`  detections   : ${JSON.stringify(result._detections)}`);
    console.log(`  ocr.found    : ${JSON.stringify(result.ocr.found)}`);
    console.log(`  ocr.detail   : ${result.ocr.detail.map((d) => `${d.list.replace(/\t/g, "\\t")}=${d.parts.map((p) => p.score.toFixed(3)).join("/")}`).join(", ")}`);
    console.log(`  faces        : ${result.faces.map((m) => `${m.name}(${m.score})`).join(", ") || "-"}`);
    console.log(`  facesWeak    : ${result.facesWeak.map((m) => `${m.name}(${m.score})`).join(", ") || "-"}`);
    console.log(`  characters   : ${result.characters.map((m) => `${m.name}[${m.source}](${m.score})`).join(", ") || "-"}`);
    console.log(`  charactersWeak: ${result.charactersWeak.map((m) => `${m.name}[${m.source}](${m.score})`).join(", ") || "-"}`);
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
