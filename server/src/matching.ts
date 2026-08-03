/**
 * 갤러리 매칭 - 얼굴/캐릭터/의상이 공통으로 쓰는 신원 판정 로직.
 *
 * 절대 임계값 하나로 자르지 않는다. 두 가지 이유:
 *  - 등록 장수가 다르면 신원마다 근방 밀도가 달라 같은 컷이 한쪽엔 과승인, 한쪽엔 과기각이 된다.
 *  - 1위와 2위가 붙어 있으면 임계값을 넘겨도 그 매칭은 무의미하다.
 * 그래서 점수는 centroid(등록 노이즈 상쇄)와 갤러리 최대값(다형성 대응)의 평균으로 내고,
 * 1위−2위 마진을 함께 요구한다.
 */

import { GalleryEntry } from "./db";

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

export interface MatchResult {
  name: string;
  score: number;
  /** 서로 다른 이름 중 2위 점수. 갤러리에 신원이 하나뿐이면 0. */
  runnerUp: number;
  margin: number;
}

/**
 * 신원 하나에 대한 점수. centroid 유사도와 개별 등록 이미지 최대 유사도의 평균.
 * 등록이 1장이면 두 값이 같으므로 그대로 그 값이 된다.
 */
function scoreEntry(query: Float32Array, entry: GalleryEntry): number {
  const centroidSim = cosineSimilarity(query, entry.centroid);
  let maxSim = -1;
  for (const v of entry.vectors) {
    const s = cosineSimilarity(query, v);
    if (s > maxSim) maxSim = s;
  }
  return (centroidSim + maxSim) / 2;
}

/**
 * 임계값과 마진을 모두 통과한 경우에만 매칭을 반환한다.
 * 통과하지 못하면 null - 틀린 이름을 내는 것보다 "모름"이 낫다.
 */
export function matchGallery(
  query: Float32Array,
  gallery: GalleryEntry[],
  threshold: number,
  minMargin: number,
  label?: string,
): MatchResult | null {
  if (gallery.length === 0) return null;

  let best = { name: "", score: -1 };
  let second = -1;
  for (const entry of gallery) {
    const s = scoreEntry(query, entry);
    if (s > best.score) {
      second = best.score;
      best = { name: entry.name, score: s };
    } else if (s > second) {
      second = s;
    }
  }

  const runnerUp = second < 0 ? 0 : second;
  const margin = best.score - runnerUp;
  const passed = best.score >= threshold && margin >= minMargin;

  if (label) {
    console.log(
      `[${label}] best="${best.name}" score=${best.score.toFixed(4)} runnerUp=${runnerUp.toFixed(4)} margin=${margin.toFixed(4)} thr=${threshold} minMargin=${minMargin} → ${passed ? "MATCH" : "REJECT"}`,
    );
  }

  return passed ? { name: best.name, score: best.score, runnerUp, margin } : null;
}
