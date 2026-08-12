/**
 * 분석 서버 - HTTP 계층.
 *
 * POST /analyze 가 네 가지를 판정한다:
 *   ocr        - searchStrings.tsv의 리스트 중 하나라도 이미지에 있는가
 *   faces      - 실사 인물 (SCRFD 랜드마크 정합 → ArcFace → 갤러리)
 *   characters - 애니 캐릭터 (WD-Tagger 제로샷 이름 + CCIP 갤러리 매칭)
 *   costumes   - 캐릭터 의상 (머리 마스킹 → WD-Tagger 의류 태그 벡터 → 갤러리)
 *
 * config.json은 요청마다 다시 읽어 재시작 없이 반영된다.
 *
 * 검색 목록은 **요청이 들고 오는 것이 정본**이다. 호출자(크롬 확장)의 키워드는 수시로
 * 바뀌는데 서버가 자기 파일만 본다면 "목록에 없어서 못 찾은 것"과 "있는데 못 읽은 것"이
 * 구분되지 않는다. multipart 필드 `searchStrings`(TSV 텍스트)로 받고, 없을 때만
 * searchStrings.tsv로 물러선다 (CLI·수동 테스트용).
 *
 * `--ocr-only`(또는 IMAGEANALYZER_OCR_ONLY=1)로 기동하면 OCR만 활성화한다. 비전 모델
 * 6개(약 772MB)를 받지도 로드하지도 않으므로 메모리가 작은 호스트에서 돌릴 수 있다.
 * 이때 faces/characters/costumes는 항상 빈 배열이다.
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
    console.warn(`Config load failed - path: ${CONFIG_PATH}, error: ${e.message} (기본값 사용)`);
    return DEFAULT_CONFIG;
  }
}

/** 요청이 목록을 주지 않았을 때만 쓰는 예비 경로 (CLI·수동 테스트). */
export function loadSearchLists() {
  if (!fs.existsSync(SEARCH_STRINGS_PATH)) return [];
  return parseSearchLists(fs.readFileSync(SEARCH_STRINGS_PATH, "utf-8"));
}

/** 비전 기능 없이 기동할지. 인자와 환경변수 어느 쪽으로도 켤 수 있다. */
export const OCR_ONLY =
  process.argv.includes("--ocr-only") || process.env.IMAGEANALYZER_OCR_ONLY === "1";

async function main() {
  console.log("Initializing ImageAnalyzer server...");

  if (OCR_ONLY) console.log("OCR-only mode - vision models are neither downloaded nor loaded");

  await ensureModelsDownloaded(MODELS_DIR, OCR_ONLY);
  initDB(DB_DIR);
  // 비전 세션을 만들지 않는다. 탐지기는 세션이 없으면 빈 배열을 돌려주도록 되어 있어
  // analyzeImage는 그대로 통과하고 faces/characters/costumes만 비게 된다.
  if (!OCR_ONLY) await loadModels(MODELS_DIR);
  await initOCR(MODELS_DIR);

  if (!OCR_ONLY) {
    console.log(
      `Gallery - face: ${countEmbeddings("face")}, character: ${countEmbeddings("character")}, costume: ${countEmbeddings("costume")}`,
    );
  }

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

      // 요청이 준 목록이 정본. 필드가 없을 때만 파일로 물러선다.
      const provided = typeof req.body?.searchStrings === "string" ? req.body.searchStrings : "";
      const searchLists = provided.trim().length > 0 ? parseSearchLists(provided) : loadSearchLists();
      const listSource = provided.trim().length > 0 ? "request" : "file";

      const result = await analyzeImage(req.file.buffer, searchLists, loadConfig());

      console.log(
        `Analyze done - ms: ${result._elapsedMs}, listSource: ${listSource}, lists: ${searchLists.length}, ocrFound: ${result.ocr.found.length}, faces: ${result.faces.length}, characters: ${result.characters.length}, costumes: ${result.costumes.length}`,
      );
      res.json(result);
    } catch (err: any) {
      console.error(`Analysis error - message: ${err.message}, at: ${err.stack?.split("\n")[1]?.trim()}`);
      res.status(500).json({ error: "Analysis failed", details: err.message });
    }
  });

  app.get("/health", (_, res) => {
    res.json({
      status: "ok",
      ocrOnly: OCR_ONLY,
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
  app.listen(PORT, () => console.log(`Server listening - url: http://localhost:${PORT}`));
}

main().catch((err) => {
  console.error("Server startup failed:", err);
  process.exit(1);
});
