/**
 * 갤러리 등록 — face / character / costume 공통 드라이버.
 *
 *   data/faces/<인물명>/*.jpg      → SCRFD 최대 얼굴 → 랜드마크 정합 → ArcFace
 *   data/characters/<캐릭터명>/*   → 애니 인물 검출 → CCIP
 *   data/costumes/<의상명>/*       → 애니 인물 검출 → 머리 마스킹 → WD-Tagger 의류 태그 벡터
 *
 * 폴더명이 곧 이름이다. 이미 등록된 이미지는 건너뛴다.
 *
 * 등록과 질의는 반드시 같은 전처리를 거쳐야 한다. 특히 얼굴은 정합 여부가 다르면
 * 갤러리와 질의가 서로 다른 분포에 놓여 유사도가 무의미해진다. 그래서 서버와
 * 동일한 inference 함수를 그대로 재사용한다.
 *
 * Usage: npx ts-node scripts/register.ts [face|character|costume|all]
 */

import path from "path";
import fs from "fs";
import { initDB, isImageRegistered, insertEmbedding, countEmbeddings, Kind, Space } from "../server/src/db";
import { ensureModelsDownloaded } from "../server/src/model-downloader";
import {
  loadModels,
  loadRaw,
  detectFaces,
  detectAnimeFaces,
  detectAnimePersons,
  detectionsInside,
  alignFace,
  extractFaceEmbedding,
  faceQualityOk,
  extractCcipEmbedding,
  runTagger,
  cropRegion,
  cropWithMask,
  Box,
  Detection,
} from "../server/src/inference";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");
const DB_DIR = path.join(PROJECT_ROOT, "db");
const SUPPORTED_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

interface KindSpec {
  kind: Kind;
  space: Space;
  dir: string;
  extract: (buffer: Buffer) => Promise<Float32Array | null>;
}

/** 여러 검출 중 가장 큰 것. 등록 이미지는 대상이 주인공이라고 가정한다. */
function largest(dets: Detection[]): Detection | null {
  if (dets.length === 0) return null;
  return dets.reduce((a, b) => (a.box[2] * a.box[3] >= b.box[2] * b.box[3] ? a : b));
}

async function extractFace(buffer: Buffer): Promise<Float32Array | null> {
  const faces = await detectFaces(buffer);
  const best = largest(faces);
  if (!best) {
    console.warn(`  ⚠️  no face detected`);
    return null;
  }
  if (!best.landmarks) {
    console.warn(`  ⚠️  no landmarks — 정합 불가라 등록하지 않는다`);
    return null;
  }
  if (!faceQualityOk(best)) {
    console.warn(`  ⚠️  low quality face - box: ${best.box.map(Math.round).join(",")}`);
    return null;
  }
  const raw = await loadRaw(buffer);
  return extractFaceEmbedding(alignFace(raw, best.landmarks));
}

async function extractCharacter(buffer: Buffer): Promise<Float32Array | null> {
  const persons = await detectAnimePersons(buffer);
  const best = largest(persons);
  // 검출이 없으면 이미 캐릭터로 크롭된 이미지라고 보고 전체를 쓴다.
  const region = best ? await cropRegion(buffer, best.box) : buffer;
  return extractCcipEmbedding(region);
}

async function extractCostume(buffer: Buffer): Promise<Float32Array | null> {
  const persons = await detectAnimePersons(buffer);
  const best = largest(persons);
  const box: Box | null = best ? best.box : null;

  const faces = await detectAnimeFaces(buffer);
  const heads = (box ? detectionsInside(box, faces) : faces).map((f) => f.box);

  // 머리·얼굴을 지워야 착용자 신원이 벡터에 섞이지 않는다.
  const region = box ? await cropWithMask(buffer, box, heads) : await cropWithMask(buffer, [0, 0, 1e9, 1e9] as Box, heads);

  const tagged = await runTagger(region);
  if (!tagged) return null;
  if (tagged.topClothing.length === 0) {
    console.warn(`  ⚠️  no clothing tags detected`);
    return null;
  }
  console.log(`  tags: ${tagged.topClothing.slice(0, 8).map((t) => `${t.name}:${t.prob.toFixed(2)}`).join(" ")}`);
  return tagged.costumeVector;
}

const SPECS: Record<string, KindSpec> = {
  face: { kind: "face", space: "arcface", dir: "faces", extract: extractFace },
  character: { kind: "character", space: "ccip", dir: "characters", extract: extractCharacter },
  costume: { kind: "costume", space: "wdtag", dir: "costumes", extract: extractCostume },
};

async function registerKind(spec: KindSpec): Promise<void> {
  const root = path.join(PROJECT_ROOT, "data", spec.dir);
  if (!fs.existsSync(root)) {
    console.log(`[${spec.kind}] directory not found - path: ${root}`);
    return;
  }

  const names = fs.readdirSync(root).filter((f) => fs.statSync(path.join(root, f)).isDirectory());
  if (names.length === 0) {
    console.log(`[${spec.kind}] no subfolders in ${root} (폴더명이 이름이 된다)`);
    return;
  }

  let registered = 0;
  let skipped = 0;
  let failed = 0;

  for (const name of names) {
    const dir = path.join(root, name);
    const images = fs.readdirSync(dir).filter((f) => SUPPORTED_EXT.has(path.extname(f).toLowerCase()));

    for (const file of images) {
      const abs = path.join(dir, file);
      const rel = path.relative(PROJECT_ROOT, abs).replace(/\\/g, "/");

      if (isImageRegistered(spec.kind, rel)) {
        skipped++;
        continue;
      }

      console.log(`[${spec.kind}] ${name} — ${file}`);
      try {
        const embedding = await spec.extract(fs.readFileSync(abs));
        if (!embedding) {
          failed++;
          continue;
        }
        insertEmbedding(spec.kind, spec.space, name, rel, embedding);
        registered++;
      } catch (e: any) {
        console.error(`  ❌ failed - path: ${rel}, error: ${e.message}`);
        failed++;
      }
    }
  }

  console.log(
    `✅ Register done - kind: ${spec.kind}, registered: ${registered}, skipped: ${skipped}, failed: ${failed}, total: ${countEmbeddings(spec.kind)}`,
  );
}

async function main() {
  const arg = (process.argv[2] || "all").toLowerCase();
  const targets = arg === "all" ? Object.values(SPECS) : [SPECS[arg]];
  if (targets.some((t) => !t)) {
    console.error(`Unknown kind: ${arg}. Use one of: face, character, costume, all`);
    process.exit(1);
  }

  await ensureModelsDownloaded(MODELS_DIR);
  initDB(DB_DIR);
  await loadModels(MODELS_DIR);

  for (const spec of targets) await registerKind(spec);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("Registration failed:", err);
    process.exit(1);
  });
