/**
 * 분석 오케스트레이션. HTTP와 분리해 CLI(scripts/analyze.ts)에서도 같은 경로를 쓴다.
 */

import path from "path";
import { getGallery, l2normalize } from "./db";
import { performOCR, SearchList, OcrResult } from "./ocr";
import { matchGallery } from "./matching";
import { collectCandidate, CandidateConfig } from "./candidates";

/** 후보 크롭이 쌓이는 곳. server/src 와 server/dist 어느 쪽에서 실행해도 프로젝트 루트를 가리킨다. */
const DATA_DIR = path.resolve(__dirname, "..", "..", "data");
import {
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
  hasArcFace,
  hasCcip,
  hasWdTagger,
  Box,
  Detection,
} from "./inference";

export interface Config {
  /**
   * face는 "실물"로 인정하는 값, faceWeak는 그 아래 보고만 하는 하한이다.
   * 사진 속 사진처럼 화질이 열화된 동일 인물은 두 값 사이에 떨어진다.
   */
  similarityThreshold: { face: number; faceWeak: number; character: number; costume: number };
  margin: { face: number; character: number; costume: number };
  ocr: { scoreThreshold: number; detScales: number[] };
  wdTagger: { characterTagThreshold: number };
  /**
   * WD-Tagger는 danbooru 로마자 태그(uehara_ayumu)를 내지만 갤러리 이름은 보통
   * 일본어 표기(上原歩夢)다. 두 경로의 출력을 같은 이름으로 통일하기 위한 매핑.
   */
  characterAliases: Record<string, string>;
  /**
   * 확정 매칭된 크롭을 data/_candidates/ 에 모아두는 설정. 갤러리에 자동으로 넣지는 않는다 —
   * 사람이 검토해서 data/faces|characters|costumes/ 로 옮긴 뒤 register를 돌린다.
   */
  candidates: CandidateConfig;
}

/**
 * 임계값은 모델마다 코사인 분포가 달라 서로 비교 불가능한 스케일 위에 있다.
 * 얼굴은 실측 분포가 명확히 갈린다 — 실물 동일 인물 0.64~0.97, 사진 속 사진 0.32,
 * 다른 인물 0.15. 그래서 0.45를 실물 경계로, 0.28을 보고 하한으로 둔다.
 * ArcFace에서 0.8 같은 값은 동일 인물조차 전부 기각한다.
 */
export const DEFAULT_CONFIG: Config = {
  similarityThreshold: { face: 0.45, faceWeak: 0.28, character: 0.82, costume: 0.55 },
  margin: { face: 0.06, character: 0.04, costume: 0.05 },
  ocr: { scoreThreshold: 0.5, detScales: [960, 1600] },
  wdTagger: { characterTagThreshold: 0.6 },
  characterAliases: {},
  candidates: { enabled: true, dedupThreshold: 0.95, maxPerName: 50 },
};

export interface Match {
  name: string;
  score: number;
  margin: number;
  box?: Box;
  source?: string;
}

export interface AnalyzeResult {
  ocr: OcrResult;
  /** 실물로 판정된 얼굴. */
  faces: Match[];
  /** faceWeak 이상 face 미만. 사진 속 사진·저화질이 여기 떨어진다. 자동 판정에 쓰지 말 것. */
  facesWeak: Match[];
  characters: Match[];
  costumes: Match[];
  _detections: { realFaces: number; animeFaces: number; animePersons: number; textRegions: number };
  _elapsedMs: number;
}

/**
 * 실사 인물. ArcFace는 5점 유사변환으로 정규 템플릿에 워프된 얼굴로만 학습되었으므로
 * 랜드마크가 없으면 임베딩이 분포 밖이 된다 — 등록하지도 질의하지도 않는다.
 */
async function recognizeFaces(
  imageBuffer: Buffer,
  faces: Detection[],
  cfg: Config,
): Promise<{ strong: Match[]; weak: Match[] }> {
  const empty = { strong: [], weak: [] };
  if (!hasArcFace() || faces.length === 0) return empty;
  const gallery = getGallery("face", "arcface");
  if (gallery.length === 0) return empty;

  const raw = await loadRaw(imageBuffer);
  const strong: Match[] = [];
  const weak: Match[] = [];

  for (const det of faces) {
    if (!det.landmarks || !faceQualityOk(det)) continue;

    const embedding = await extractFaceEmbedding(alignFace(raw, det.landmarks));
    if (!embedding) continue;

    // 하한으로 한 번만 매칭하고 점수로 두 단계를 가른다. 두 번 돌릴 필요가 없다.
    const best = matchGallery(embedding, gallery, cfg.similarityThreshold.faceWeak, cfg.margin.face, "face");
    if (!best) continue;

    const match: Match = {
      name: best.name,
      score: +best.score.toFixed(4),
      margin: +best.margin.toFixed(4),
      box: det.box.map(Math.round) as Box,
    };
    if (best.score >= cfg.similarityThreshold.face) {
      strong.push(match);
      // 확정된 것만 후보로 모은다. 등록 시 SCRFD가 다시 얼굴을 찾아야 하므로
      // 박스를 60% 넓혀 여유를 준 크롭을 남긴다.
      const [x, y, w, h] = det.box;
      const pad = 0.6;
      await collectCandidate({
        dataDir: DATA_DIR,
        kind: "face",
        space: "arcface",
        name: best.name,
        embedding,
        crop: await cropRegion(imageBuffer, [x - w * pad / 2, y - h * pad / 2, w * (1 + pad), h * (1 + pad)]),
        score: best.score,
        source: "arcface",
        config: cfg.candidates,
      });
    } else weak.push(match);
  }
  return { strong, weak };
}

/**
 * 캐릭터. **등록한 캐릭터만 보고한다.**
 *
 * 신원 판정의 주축은 CCIP다 — 갤러리 기반이라 등록만 되어 있으면 어떤 캐릭터든 잡고,
 * 서로 다른 작가·화풍의 같은 캐릭터를 positive로 학습해서 등록 1장으로도 의상·구도가
 * 달라진 이미지를 매칭한다(실측: 1장 등록 → 다른 의상 0.85, 다른 교복 0.87).
 *
 * WD-Tagger는 보조 신호일 뿐이다. 어휘가 약 2,751종으로 고정돼 있어 신작 캐릭터를
 * 아예 모른다 — 蓮ノ空 학원 캐릭터는 12명 전원이 어휘에 없다. 그래서 태거가 낸 이름은
 * 갤러리에 등록된 것만 통과시킨다.
 */
async function recognizeCharacters(imageBuffer: Buffer, persons: Detection[], cfg: Config): Promise<Match[]> {
  const ccipGallery = hasCcip() ? getGallery("character", "ccip") : [];
  const registered = new Set(ccipGallery.map((g) => g.name));
  if (registered.size === 0) return [];

  const matches: Match[] = [];
  // 캐릭터 영역이 없으면 이미지 전체를 하나의 영역으로 본다.
  const regions: (Box | null)[] = persons.length > 0 ? persons.map((p) => p.box) : [null];

  for (const box of regions) {
    const buf = box ? await cropRegion(imageBuffer, box) : imageBuffer;

    // 이 영역에서 확정된 이름. 태거와 CCIP 중 하나만 확정해도 후보 수집 대상이 된다.
    let confirmed: { name: string; score: number; source: string } | null = null;

    if (hasWdTagger()) {
      const tagged = await runTagger(buf, cfg.wdTagger.characterTagThreshold);
      for (const c of tagged?.characters ?? []) {
        const name = cfg.characterAliases[c.name] ?? c.name;
        if (!registered.has(name)) continue;
        if (!matches.some((m) => m.name === name && m.source === "tagger")) {
          matches.push({ name, score: +c.prob.toFixed(4), margin: 0, box: box ?? undefined, source: "tagger" });
        }
        if (!confirmed) confirmed = { name, score: c.prob, source: "tagger" };
      }
    }

    const embedding = ccipGallery.length > 0 ? await extractCcipEmbedding(buf) : null;
    if (embedding) {
      const best = matchGallery(embedding, ccipGallery, cfg.similarityThreshold.character, cfg.margin.character, "character");
      if (best) {
        if (!matches.some((m) => m.name === best.name && m.source === "ccip")) {
          matches.push({ name: best.name, score: +best.score.toFixed(4), margin: +best.margin.toFixed(4), box: box ?? undefined, source: "ccip" });
        }
        confirmed = { name: best.name, score: best.score, source: "ccip" };
      }
    }

    // 태거만 확정하고 CCIP는 놓친 크롭이 갤러리에 가장 필요한 이미지다 —
    // 그 화풍·구도를 CCIP가 아직 못 잡고 있다는 뜻이므로.
    if (confirmed && embedding) {
      await collectCandidate({
        dataDir: DATA_DIR,
        kind: "character",
        space: "ccip",
        name: confirmed.name,
        embedding,
        crop: buf,
        score: confirmed.score,
        source: confirmed.source,
        config: cfg.candidates,
      });
    }
  }

  return matches;
}

/**
 * 캐릭터 의상.
 *
 * 요구사항이 "다른 그림체로 그려져도, 다른 캐릭터가 입고 있어도 같은 의상으로 발견"이라
 * CLIP·CCIP 임베딩은 쓸 수 없다 — 전자는 화풍을, 후자는 착용자 신원을 함께 인코딩하므로
 * 둘 중 하나만 바뀌어도 거리가 벌어진다. 대신 WD-Tagger의 의류 태그 확률 벡터를 쓴다.
 * 태그는 의미 단위라 화풍에 불변이고, 옷만 기술하므로 착용자에도 불변이다.
 * 머리·얼굴은 마스킹해 신원 단서를 제거한다.
 */
async function recognizeCostumes(
  imageBuffer: Buffer,
  persons: Detection[],
  animeFaces: Detection[],
  cfg: Config,
): Promise<Match[]> {
  if (!hasWdTagger()) return [];
  const gallery = getGallery("costume", "wdtag");
  if (gallery.length === 0 || persons.length === 0) return [];

  const matches: Match[] = [];
  for (const person of persons) {
    const heads = detectionsInside(person.box, animeFaces).map((f) => f.box);
    const region = await cropWithMask(imageBuffer, person.box, heads);

    const tagged = await runTagger(region, cfg.wdTagger.characterTagThreshold);
    if (!tagged) continue;

    const best = matchGallery(
      l2normalize(tagged.costumeVector),
      gallery,
      cfg.similarityThreshold.costume,
      cfg.margin.costume,
      "costume",
    );
    if (best) {
      matches.push({
        name: best.name,
        score: +best.score.toFixed(4),
        margin: +best.margin.toFixed(4),
        box: person.box.map(Math.round) as Box,
        source: tagged.topClothing.slice(0, 6).map((t) => t.name).join(","),
      });
      await collectCandidate({
        dataDir: DATA_DIR,
        kind: "costume",
        space: "wdtag",
        name: best.name,
        embedding: tagged.costumeVector,
        crop: region,
        score: best.score,
        source: "wdtag",
        config: cfg.candidates,
      });
    }
  }
  return matches;
}

export async function analyzeImage(
  imageBuffer: Buffer,
  searchLists: SearchList[],
  cfg: Config,
): Promise<AnalyzeResult> {
  const started = Date.now();

  const [ocr, realFaces, animeFaces, animePersons] = await Promise.all([
    performOCR(imageBuffer, searchLists, cfg.ocr),
    detectFaces(imageBuffer),
    detectAnimeFaces(imageBuffer),
    detectAnimePersons(imageBuffer),
  ]);

  const faces = await recognizeFaces(imageBuffer, realFaces, cfg);
  const characters = await recognizeCharacters(imageBuffer, animePersons, cfg);
  const costumes = await recognizeCostumes(imageBuffer, animePersons, animeFaces, cfg);

  return {
    ocr,
    faces: faces.strong,
    facesWeak: faces.weak,
    characters,
    costumes,
    _detections: {
      realFaces: realFaces.length,
      animeFaces: animeFaces.length,
      animePersons: animePersons.length,
      textRegions: ocr.regions,
    },
    _elapsedMs: Date.now() - started,
  };
}
