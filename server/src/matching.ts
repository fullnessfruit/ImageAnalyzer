/**
 * 갤러리 매칭 - 얼굴/캐릭터/의상이 공통으로 쓰는 신원 판정 로직.
 *
 * 절대 임계값 하나로 자르지 않는다. 두 가지 이유:
 *  - 등록 장수가 다르면 신원마다 근방 밀도가 달라 같은 컷이 한쪽엔 과승인, 한쪽엔 과기각이 된다.
 *  - 1위와 2위가 붙어 있으면 임계값을 넘겨도 그 매칭은 무의미하다.
 * 얼굴과 캐릭터는 centroid(등록 노이즈 상쇄)와 갤러리 최대값(다형성 대응)의 평균으로 낸다.
 * 한 이름 아래 서로 다른 의상이 들어갈 수 있는 의상은 최대값을 쓴다. 모든 과제에
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
 * 얼굴·캐릭터의 hybrid는 centroid와 가장 가까운 등록 이미지의 평균이다.
 * max는 서로 다른 의상이 같은 이름 아래 있을 때 중심점이 실제 어느 의상과도
 * 닮지 않는 문제를 피한다. 등록이 한 장이면 두 방식의 점수는 같다.
 */
export type ScoreMode = "hybrid" | "max";

function scoreEntry(query: Float32Array, entry: GalleryEntry, mode: ScoreMode): number {
  let maxSim = -1;
  for (const v of entry.vectors) {
    const s = cosineSimilarity(query, v);
    if (s > maxSim) maxSim = s;
  }
  if (mode === "max") return maxSim;
  const centroidSim = cosineSimilarity(query, entry.centroid);
  return (centroidSim + maxSim) / 2;
}

/**
 * 최고 점수 신원을 먼저 고른 다음 그 신원의 임계값과 마진을 확인한다.
 * 임계값 미달 시 2위로 대체하지 않는다. 이름별 임계값은 얼굴의 약한
 * 결과에 사용하고, 나머지 과제는 공통 숫자 임계값을 사용한다.
 * 통과하지 못하면 null - 틀린 이름을 내는 것보다 "모름"이 낫다.
 */
export function matchGallery(
  query: Float32Array,
  gallery: GalleryEntry[],
  threshold: number | ((name: string) => number),
  minMargin: number,
  label?: string,
  scoreMode: ScoreMode = "hybrid",
): MatchResult | null {
  if (gallery.length === 0) return null;

  let best = { name: "", score: -1 };
  let second = -1;
  for (const entry of gallery) {
    const s = scoreEntry(query, entry, scoreMode);
    if (s > best.score) {
      second = best.score;
      best = { name: entry.name, score: s };
    } else if (s > second) {
      second = s;
    }
  }

  const runnerUp = second < 0 ? 0 : second;
  const margin = best.score - runnerUp;
  const appliedThreshold = typeof threshold === "number" ? threshold : threshold(best.name);
  const passed = best.score >= appliedThreshold && margin >= minMargin;

  if (label) {
    console.log(
      `[${label}] best="${best.name}" score=${best.score.toFixed(4)} runnerUp=${runnerUp.toFixed(4)} margin=${margin.toFixed(4)} thr=${appliedThreshold} minMargin=${minMargin} → ${passed ? "MATCH" : "REJECT"}`,
    );
  }

  return passed ? { name: best.name, score: best.score, runnerUp, margin } : null;
}
