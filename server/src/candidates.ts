/**
 * 후보 수집 - 확정 매칭된 크롭을 나중에 검토해서 등록할 수 있도록 파일로 남긴다.
 *
 * 갤러리에 자동으로 넣지 않는다. 자동 등록은 잘못 들어간 항목이 조용히 이후 매칭을 바꾸고,
 * 되돌리려면 어느 항목이 잘못됐는지 알아야 하는데 분석 중 잘라낸 크롭은 디스크에 없어
 * 눈으로 확인할 수가 없다. 결국 전부 지우는 맹목 롤백만 남는다.
 * 파일로 남겨두면 사람이 추려서 data/ 로 옮기고, 갤러리에는 승인된 것만 들어간다.
 *
 * 매칭되지 않은 크롭은 저장하지 않는다 - 확정된 신원의 갤러리를 넓히는 것이 목적이다.
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";
import sharp from "sharp";
import {
  Kind,
  Space,
  insertCandidate,
  getCandidateVectors,
  countCandidates,
  l2normalize,
} from "./db";
import { cosineSimilarity } from "./matching";

export interface CandidateConfig {
  enabled: boolean;
  /** 이 이상 닮은 후보가 이미 있으면 저장하지 않는다. 같은 장면에서 수백 장이 쌓이는 것을 막는다. */
  dedupThreshold: number;
  /** 신원당 후보 상한. 검토 가능한 분량을 넘지 않게 한다. */
  maxPerName: number;
}

export const CANDIDATES_DIR = "_candidates";

/** 파일명으로 쓸 수 없는 문자만 치환한다. 일본어·한글 이름은 그대로 둔다. */
function safeName(name: string): string {
  return name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").trim() || "_";
}

/**
 * 확정 매칭된 크롭을 후보로 저장한다.
 * 이미 비슷한 후보가 있거나 상한에 도달했으면 아무것도 하지 않는다.
 */
export async function collectCandidate(opts: {
  dataDir: string;
  kind: Kind;
  space: Space;
  name: string;
  embedding: Float32Array;
  crop: Buffer;
  score: number;
  source: string;
  config: CandidateConfig;
}): Promise<void> {
  const { dataDir, kind, space, name, embedding, crop, score, source, config } = opts;
  if (!config.enabled) return;

  if (countCandidates(kind, name) >= config.maxPerName) return;

  const normalized = l2normalize(embedding);
  for (const existing of getCandidateVectors(kind, name)) {
    if (cosineSimilarity(normalized, existing) >= config.dedupThreshold) return;
  }

  const dir = path.join(dataDir, CANDIDATES_DIR, kind, safeName(name));
  fs.mkdirSync(dir, { recursive: true });

  const hash = crypto.createHash("sha1").update(crop).digest("hex").slice(0, 12);
  const file = path.join(dir, `${score.toFixed(3)}-${hash}.jpg`);
  if (fs.existsSync(file)) return;

  await sharp(crop).jpeg({ quality: 92 }).toFile(file);
  insertCandidate(kind, space, name, path.relative(dataDir, file).replace(/\\/g, "/"), embedding, score, source);

  console.log(`Candidate saved - kind: ${kind}, name: ${name}, score: ${score.toFixed(4)}, file: ${path.basename(file)}, source: ${source}`);
}
