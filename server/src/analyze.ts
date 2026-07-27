/**
 * 분석 오케스트레이션. HTTP와 분리해 CLI(scripts/analyze.ts)에서도 같은 경로를 쓴다.
 */

import { getGallery, l2normalize } from "./db";
import { performOCR, SearchList, OcrResult } from "./ocr";
import { matchGallery } from "./matching";
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
  similarityThreshold: { face: number; character: number; costume: number };
  margin: { face: number; character: number; costume: number };
  ocr: { scoreThreshold: number; detScales: number[] };
  wdTagger: { characterTagThreshold: number };
  /**
   * WD-Tagger는 danbooru 로마자 태그(uehara_ayumu)를 내지만 갤러리 이름은 보통
   * 일본어 표기(上原歩夢)다. 두 경로의 출력을 같은 이름으로 통일하기 위한 매핑.
   */
  characterAliases: Record<string, string>;
}

/**
 * 임계값은 모델마다 코사인 분포가 달라 서로 비교 불가능한 스케일 위에 있다.
 * face 0.28은 실측 근거가 있다 — 동일 인물 0.64~0.79, 화질 열화본 0.32,
 * 다른 인물 0.15. ArcFace에서 0.8 같은 값은 동일 인물조차 전부 기각한다.
 */
export const DEFAULT_CONFIG: Config = {
  similarityThreshold: { face: 0.28, character: 0.82, costume: 0.55 },
  margin: { face: 0.06, character: 0.04, costume: 0.05 },
  ocr: { scoreThreshold: 0.5, detScales: [960, 1600] },
  wdTagger: { characterTagThreshold: 0.6 },
  characterAliases: {},
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
  faces: Match[];
  characters: Match[];
  costumes: Match[];
  _detections: { realFaces: number; animeFaces: number; animePersons: number; textRegions: number };
  _elapsedMs: number;
}

/**
 * 실사 인물. ArcFace는 5점 유사변환으로 정규 템플릿에 워프된 얼굴로만 학습되었으므로
 * 랜드마크가 없으면 임베딩이 분포 밖이 된다 — 등록하지도 질의하지도 않는다.
 */
async function recognizeFaces(imageBuffer: Buffer, faces: Detection[], cfg: Config): Promise<Match[]> {
  if (!hasArcFace() || faces.length === 0) return [];
  const gallery = getGallery("face", "arcface");
  if (gallery.length === 0) return [];

  const raw = await loadRaw(imageBuffer);
  const matches: Match[] = [];

  for (const det of faces) {
    if (!det.landmarks || !faceQualityOk(det)) continue;

    const embedding = await extractFaceEmbedding(alignFace(raw, det.landmarks));
    if (!embedding) continue;

    const best = matchGallery(embedding, gallery, cfg.similarityThreshold.face, cfg.margin.face, "face");
    if (best) {
      matches.push({ name: best.name, score: +best.score.toFixed(4), margin: +best.margin.toFixed(4), box: det.box.map(Math.round) as Box });
    }
  }
  return matches;
}

/**
 * 캐릭터. 실패 지점이 다른 두 신호를 겹친다.
 *  - WD-Tagger: 세계 지식 기반. 등록 0장으로 알려진 캐릭터 이름이 나오고 화풍 변화에 강하다.
 *  - CCIP: 갤러리 기반. 태거가 모르는 캐릭터를 1~2장 등록으로 잡는다.
 */
async function recognizeCharacters(imageBuffer: Buffer, persons: Detection[], cfg: Config): Promise<Match[]> {
  const matches: Match[] = [];
  // 캐릭터 영역이 없으면 이미지 전체를 하나의 영역으로 본다.
  const regions: (Box | null)[] = persons.length > 0 ? persons.map((p) => p.box) : [null];
  const ccipGallery = hasCcip() ? getGallery("character", "ccip") : [];

  for (const box of regions) {
    const buf = box ? await cropRegion(imageBuffer, box) : imageBuffer;

    if (hasWdTagger()) {
      const tagged = await runTagger(buf, cfg.wdTagger.characterTagThreshold);
      for (const c of tagged?.characters ?? []) {
        const name = cfg.characterAliases[c.name] ?? c.name;
        if (!matches.some((m) => m.name === name && m.source === "tagger")) {
          matches.push({ name, score: +c.prob.toFixed(4), margin: 0, box: box ?? undefined, source: "tagger" });
        }
      }
    }

    if (ccipGallery.length > 0) {
      const embedding = await extractCcipEmbedding(buf);
      if (embedding) {
        const best = matchGallery(embedding, ccipGallery, cfg.similarityThreshold.character, cfg.margin.character, "character");
        if (best && !matches.some((m) => m.name === best.name && m.source === "ccip")) {
          matches.push({ name: best.name, score: +best.score.toFixed(4), margin: +best.margin.toFixed(4), box: box ?? undefined, source: "ccip" });
        }
      }
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
    faces,
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
