/**
 * 분석 서버 — HTTP 계층.
 *
 * POST /analyze 가 네 가지를 판정한다:
 *   ocr        — searchStrings.tsv의 리스트 중 하나라도 이미지에 있는가
 *   faces      — 실사 인물 (SCRFD 랜드마크 정합 → ArcFace → 갤러리)
 *   characters — 애니 캐릭터 (WD-Tagger 제로샷 이름 + CCIP 갤러리 매칭)
 *   costumes   — 캐릭터 의상 (머리 마스킹 → WD-Tagger 의류 태그 벡터 → 갤러리)
 *
 * config.json과 searchStrings.tsv는 요청마다 다시 읽어 재시작 없이 반영된다.
 */

import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";

import { initDB, countEmbeddings } from "./db";
import { ensureModelsDownloaded } from "./model-downloader";
import { initOCR, parseSearchLists, isOcrReady } from "./ocr";
import { loadModels, hasArcFace, hasCcip, hasWdTagger } from "./inference";
import { analyzeImage, Config, DEFAULT_CONFIG } from "./analyze";

const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");
const DB_DIR = path.join(PROJECT_ROOT, "db");
const CONFIG_PATH = path.join(PROJECT_ROOT, "config.json");
const SEARCH_STRINGS_PATH = path.join(PROJECT_ROOT, "searchStrings.tsv");

/** 설정 파일이 없거나 깨져도 서버는 계속 동작해야 한다. */
export function loadConfig(): Config {
  try {
    const raw = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));
    return {
      similarityThreshold: { ...DEFAULT_CONFIG.similarityThreshold, ...(raw.similarityThreshold ?? {}) },
      margin: { ...DEFAULT_CONFIG.margin, ...(raw.margin ?? {}) },
      ocr: { ...DEFAULT_CONFIG.ocr, ...(raw.ocr ?? {}) },
      wdTagger: { ...DEFAULT_CONFIG.wdTagger, ...(raw.wdTagger ?? {}) },
      characterAliases: { ...DEFAULT_CONFIG.characterAliases, ...(raw.characterAliases ?? {}) },
      candidates: { ...DEFAULT_CONFIG.candidates, ...(raw.candidates ?? {}) },
    };
  } catch (e: any) {
    console.warn(`⚠️  Config load failed - path: ${CONFIG_PATH}, error: ${e.message} (기본값 사용)`);
    return DEFAULT_CONFIG;
  }
}

export function loadSearchLists() {
  if (!fs.existsSync(SEARCH_STRINGS_PATH)) return [];
  return parseSearchLists(fs.readFileSync(SEARCH_STRINGS_PATH, "utf-8"));
}

async function main() {
  console.log("Initializing ImageAnalyzer server...");

  await ensureModelsDownloaded(MODELS_DIR);
  initDB(DB_DIR);
  await loadModels(MODELS_DIR);
  await initOCR(MODELS_DIR);

  console.log(
    `📊 Gallery - face: ${countEmbeddings("face")}, character: ${countEmbeddings("character")}, costume: ${countEmbeddings("costume")}`,
  );

  const app = express();
  app.use((_, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    res.header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    next();
  });

  const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 32 * 1024 * 1024 } });

  app.post("/analyze", upload.single("image"), async (req, res) => {
    try {
      if (!req.file) {
        res.status(400).json({ error: "No image file provided. Use field name 'image'." });
        return;
      }

      const result = await analyzeImage(req.file.buffer, loadSearchLists(), loadConfig());

      console.log(
        `✅ Analyze done - ms: ${result._elapsedMs}, ocrFound: ${result.ocr.found.length}, faces: ${result.faces.length}, characters: ${result.characters.length}, costumes: ${result.costumes.length}`,
      );
      res.json(result);
    } catch (err: any) {
      console.error(`❌ Analysis error - message: ${err.message}, at: ${err.stack?.split("\n")[1]?.trim()}`);
      res.status(500).json({ error: "Analysis failed", details: err.message });
    }
  });

  app.get("/health", (_, res) => {
    res.json({
      status: "ok",
      ocr: isOcrReady(),
      arcface: hasArcFace(),
      ccip: hasCcip(),
      wdTagger: hasWdTagger(),
      gallery: {
        face: countEmbeddings("face"),
        character: countEmbeddings("character"),
        costume: countEmbeddings("costume"),
      },
    });
  });

  const PORT = 3000;
  app.listen(PORT, () => console.log(`✅ Server listening - url: http://localhost:${PORT}`));
}

main().catch((err) => {
  console.error("Server startup failed:", err);
  process.exit(1);
});
