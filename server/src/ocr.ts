/**
 * OCR - "검색 목록의 문자열이 이미지에 있는가"만 판정한다.
 *
 * 전문(全文)을 조립하지 않는다. 요구사항이 목록 매칭뿐이라 읽기 순서 복원, 라인 결합,
 * 언어 폴백 판정이 전부 불필요하다.
 *
 * 핵심 설계 - greedy 디코딩 후 부분문자열 비교를 하지 않는다.
 * greedy는 타임스텝마다 argmax를 취하므로 정답 후보 집합을 전혀 활용하지 못하고,
 * 정확 일치 확률이 문자당 정확도의 L제곱으로 떨어진다(5글자면 p=0.95라도 0.77).
 * 대신 CTC 격자 위에서 후보 문자열의 사후확률을 직접 계산한다. 탐색 공간이
 * |사전|^T 에서 검색어 개수로 붕괴하므로, 한 글자가 2위로 밀려도 매칭이 살아남는다.
 *
 * 매칭 규칙: 한 줄 = 하나의 리스트. 줄 안의 탭 구분 파트는 모두 존재해야 한다(AND).
 * 줄끼리는 OR - 한 리스트라도 일치하면 성공.
 */

import * as ort from "onnxruntime-node";
import sharp from "sharp";
import fs from "fs";
import { getModelPath } from "./model-downloader";

// ============================================================
// 상태
// ============================================================

let detSession: ort.InferenceSession | null = null;

interface RecModel {
  session: ort.InferenceSession;
  /** 문자 → class 인덱스(1-based). class 0은 blank. */
  charIndex: Map<string, number>;
}
const recModels = new Map<RecLang, RecModel>();

/**
 * 검색어를 표현할 수 있는 모델은 전부 채점에 쓴다. 표기로 모델을 고르지 않는다.
 *
 * 이름이 오해를 부르지만 **"ch"는 중국어 전용이 아니라 CJK+라틴 통합 사전**이다.
 * 18,383자에 한자·히라가나·가타카나·라틴·숫자가 모두 들어 있다.
 * 다만 **사전이 덮는 것과 모델이 잘 읽는 것은 다르다** - 학습 언어는 중국어라
 * 일본어 표기를 간체자로 끌어당긴다(亜→亚). 그래서 표현 가능한 모델을 전부 돌린다.
 * "ko"(11,945자)에는 한자와 가나가 아예 없어 한글이 섞인 검색어만 이쪽이 필요하다.
 * 자종별 내역은 Document.md "검색어를 표현할 수 있는 모델은 전부 채점한다" 참조.
 */
export type RecLang = "ch" | "ko";

/** 로그용 사전 커버리지 설명. 모델을 추가하면 여기에도 한 줄 넣는다. */
const REC_COVERAGE: Record<RecLang, string> = {
  ch: "kanji/kana/latin/digits (zh-trained)",
  ko: "hangul/latin/digits",
};

export function isOcrReady(): boolean {
  return detSession !== null && recModels.size > 0;
}

export async function initOCR(modelsDir: string): Promise<void> {
  const detPath = getModelPath(modelsDir, "ocr-det");
  if (!fs.existsSync(detPath)) {
    console.warn(`OCR detection model missing - path: ${detPath} (OCR 비활성)`);
    return;
  }
  detSession = await ort.InferenceSession.create(detPath, { intraOpNumThreads: 4 });

  for (const [lang, model, dict] of [
    ["ch", "ocr-rec-ch", "ocr-dict-ch"],
    ["ko", "ocr-rec-ko", "ocr-dict-ko"],
  ] as const) {
    const recPath = getModelPath(modelsDir, model);
    const dictPath = getModelPath(modelsDir, dict);
    if (!fs.existsSync(recPath) || !fs.existsSync(dictPath)) {
      console.warn(`OCR rec model missing - lang: ${lang}`);
      continue;
    }

    const session = await ort.InferenceSession.create(recPath, { intraOpNumThreads: 4 });
    // PaddleOCR은 use_space_char=True로 내보내면 사전 끝에 공백이 붙는다.
    // 클래스 수 = 사전 + 공백 + blank 이므로 공백을 명시적으로 추가해야 인덱스가 어긋나지 않는다.
    const chars = fs
      .readFileSync(dictPath, "utf-8")
      .split("\n")
      .map((l) => l.replace(/\r$/, ""))
      .filter((l) => l.length > 0);
    chars.push(" ");

    const charIndex = new Map<string, number>();
    for (let i = 0; i < chars.length; i++) if (!charIndex.has(chars[i])) charIndex.set(chars[i], i + 1);

    recModels.set(lang, { session, charIndex });
    // "lang: ch"가 "중국어 전용"으로 읽히지만 아니다 - RecLang 주석 참조.
    console.log(`OCR rec loaded - lang: ${lang}, covers: ${REC_COVERAGE[lang]}, classes: ${chars.length + 1}`);
  }

  console.log(`OCR ready - det: yes, rec: ${[...recModels.keys()].map((l) => `${l}(${REC_COVERAGE[l]})`).join(", ")}`);
}

// ============================================================
// 검색 목록
// ============================================================

export interface SearchList {
  /** 파일에 적힌 원본 줄. 응답에 그대로 돌려준다. */
  raw: string;
  /** 탭으로 나뉜 파트들. 모두 존재해야 이 리스트가 일치한다. */
  parts: string[];
}

export function parseSearchLists(tsv: string): SearchList[] {
  return tsv
    .split("\n")
    .map((line) => line.replace(/\r$/, ""))
    .filter((line) => line.trim().length > 0)
    .map((raw) => ({ raw, parts: raw.split("\t").map((p) => p.trim()).filter((p) => p.length > 0) }))
    .filter((l) => l.parts.length > 0);
}

/**
 * OCR이 상습적으로 뒤바꾸는 문자 묶음. 글자 모양이 거의 같아 인식기가 자주 헷갈린다.
 * 실측 사례: 虹ヶ咲 → "虹久咲", 大西亜玖璃 → "大西亚玖璃".
 */
const CONFUSION_GROUPS = [
  "ヶケヵカ力久ク个",
  "ロ口0Oo〇",
  "エ工ェ",
  "二ニ",
  "一ー－‐–-ｰ",
  "卜ト",
  "タ夕",
  "ハ八",
  "リ川",
  "ミ三",
  "亜亞亚",
  "シツ",
  "ソン",
  "ウゥ",
  "ヌス",
  "ヲテ",
  "巳已己",
  "土士",
  "未末",
  "干千",
];

const CONFUSION_MAP = new Map<string, string[]>();
for (const group of CONFUSION_GROUPS) {
  for (const ch of group) {
    const existing = CONFUSION_MAP.get(ch) ?? [];
    CONFUSION_MAP.set(ch, [...new Set([...existing, ...group])]);
  }
}

/**
 * 한 위치에서 받아들일 문자들. 표기 흔들림(가나 종류, 전각/반각, 대소문자)과
 * OCR 혼동을 모두 여기서 흡수한다.
 *
 * 문자열 변형을 조합으로 늘리지 않는 이유 - 혼동 문자가 여러 위치에 있으면 변형 수가
 * 지수로 늘어난다. 격자 스코어링에서 위치마다 대체 문자들의 최대 확률을 쓰면 같은 효과를
 * 선형 비용으로 얻는다.
 */
function alternativesFor(ch: string): string[] {
  const alts = new Set<string>([ch]);

  const code = ch.charCodeAt(0);
  if (code >= 0x30a1 && code <= 0x30f6) alts.add(String.fromCharCode(code - 0x60)); // 가타카나 → 히라가나
  if (code >= 0x3041 && code <= 0x3096) alts.add(String.fromCharCode(code + 0x60)); // 히라가나 → 가타카나
  if (code >= 0xff01 && code <= 0xff5e) alts.add(String.fromCharCode(code - 0xfee0)); // 전각 → 반각
  if (code >= 0x21 && code <= 0x7e) alts.add(String.fromCharCode(code + 0xfee0)); // 반각 → 전각

  if (/[a-z]/.test(ch)) alts.add(ch.toUpperCase());
  if (/[A-Z]/.test(ch)) alts.add(ch.toLowerCase());

  for (const c of [...alts]) {
    for (const conf of CONFUSION_MAP.get(c) ?? []) alts.add(conf);
  }

  return [...alts];
}

// ============================================================
// 텍스트 영역 검출 (DBNet)
// ============================================================

const DET_THRESH = 0.3;
/**
 * 박스 점수는 바운딩 박스 영역의 확률 평균으로 낸다(PaddleOCR box_score_fast와 동일).
 * 키워드 스포팅에서는 재현율이 정밀도보다 훨씬 중요하다 - 잉여 영역은 어떤 검색어와도
 * 매칭되지 않고 사라질 뿐이지만, 놓친 영역은 복구할 방법이 없다.
 */
const DET_BOX_THRESH = 0.4;
const DET_MIN_AREA = 12;
const DET_UNCLIP_RATIO = 1.6;

interface TextBox {
  box: [number, number, number, number];
  score: number;
}

/**
 * 디코딩된 원본. sharp는 호출마다 소스를 다시 디코딩하므로 영역이 수십 개일 때
 * JPEG 디코딩만 수십 번 반복된다. 한 번 raw로 풀어두고 모든 크롭이 이것을 공유한다.
 */
interface RawSrc {
  buf: Buffer;
  width: number;
  height: number;
}

function fromRaw(src: RawSrc): sharp.Sharp {
  return sharp(src.buf, { raw: { width: src.width, height: src.height, channels: 3 } });
}

async function detectAtScale(src: RawSrc, limitSide: number): Promise<TextBox[]> {
  if (!detSession) return [];

  const origW = src.width;
  const origH = src.height;

  const ratio = Math.min(limitSide / Math.max(origW, origH), 1.0);
  const newW = Math.max(32, Math.round((origW * ratio) / 32) * 32);
  const newH = Math.max(32, Math.round((origH * ratio) / 32) * 32);
  const scaleX = origW / newW;
  const scaleY = origH / newH;

  const { data } = await fromRaw(src)
    .resize(newW, newH, { fit: "fill" })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const pixels = newH * newW;
  const input = new Float32Array(3 * pixels);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < pixels; i++) {
      input[c * pixels + i] = (data[i * 3 + c] / 255.0 - mean[c]) / std[c];
    }
  }

  const results = await detSession.run({
    [detSession.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, newH, newW]),
  });
  // 출력명이 sigmoid_0.tmp_0 - 이미 확률맵이다. 추가 활성화가 필요 없다.
  const prob = results[detSession.outputNames[0]].data as Float32Array;

  const binary = new Uint8Array(pixels);
  for (let i = 0; i < pixels; i++) binary[i] = prob[i] >= DET_THRESH ? 1 : 0;

  // 연결 컴포넌트 (스택 DFS)
  const labels = new Int32Array(pixels);
  const stack: number[] = [];
  let numLabels = 0;
  for (let i = 0; i < pixels; i++) {
    if (binary[i] !== 1 || labels[i] !== 0) continue;
    numLabels++;
    labels[i] = numLabels;
    stack.push(i);
    while (stack.length > 0) {
      const pos = stack.pop()!;
      const px = pos % newW;
      const py = (pos - px) / newW;
      if (py > 0 && binary[pos - newW] === 1 && labels[pos - newW] === 0) { labels[pos - newW] = numLabels; stack.push(pos - newW); }
      if (py < newH - 1 && binary[pos + newW] === 1 && labels[pos + newW] === 0) { labels[pos + newW] = numLabels; stack.push(pos + newW); }
      if (px > 0 && binary[pos - 1] === 1 && labels[pos - 1] === 0) { labels[pos - 1] = numLabels; stack.push(pos - 1); }
      if (px < newW - 1 && binary[pos + 1] === 1 && labels[pos + 1] === 0) { labels[pos + 1] = numLabels; stack.push(pos + 1); }
    }
  }

  const comps = Array.from({ length: numLabels }, () => ({
    minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity, area: 0,
  }));
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const label = labels[y * newW + x];
      if (label === 0) continue;
      const c = comps[label - 1];
      if (x < c.minX) c.minX = x;
      if (y < c.minY) c.minY = y;
      if (x > c.maxX) c.maxX = x;
      if (y > c.maxY) c.maxY = y;
      c.area++;
    }
  }

  const boxes: TextBox[] = [];
  for (const c of comps) {
    if (c.area < DET_MIN_AREA) continue;

    const bw = c.maxX - c.minX + 1;
    const bh = c.maxY - c.minY + 1;

    // 박스 영역 확률 평균
    let sum = 0;
    for (let y = c.minY; y <= c.maxY; y++) {
      for (let x = c.minX; x <= c.maxX; x++) sum += prob[y * newW + x];
    }
    const score = sum / (bw * bh);
    if (score < DET_BOX_THRESH) continue;

    // DBNet은 축소된 폴리곤을 출력한다. 복원은 모든 변을 같은 거리만큼 바깥으로 미는
    // 등방 확장이어야 한다. 박스 크기에 비례해 늘리면 가로로만 과확장되어 옆 글자를 삼킨다.
    const offset = (bw * bh * DET_UNCLIP_RATIO) / (2 * (bw + bh));
    const x0 = Math.max(0, (c.minX - offset) * scaleX);
    const y0 = Math.max(0, (c.minY - offset) * scaleY);
    const x1 = Math.min(origW, (c.maxX + 1 + offset) * scaleX);
    const y1 = Math.min(origH, (c.maxY + 1 + offset) * scaleY);
    if (x1 - x0 < 2 || y1 - y0 < 2) continue;

    boxes.push({ box: [x0, y0, x1 - x0, y1 - y0], score });
  }

  return boxes;
}

function boxIou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * 여러 해상도에서 검출해 합친다. 단일 960px 패스는 1920px 스크린샷의 작은 글자를
 * 검출기 유효 최소 크기 아래로 축소시켜 통째로 놓친다.
 */
async function detectTextRegions(src: RawSrc, scales: number[]): Promise<TextBox[]> {
  // 원본보다 큰 스케일은 1.0으로 클램프되므로 서로 다른 스케일이 같은 크기로 수렴한다.
  // 작은 이미지에서 같은 검출을 두 번 돌리지 않도록 실효 크기 기준으로 중복을 제거한다.
  const maxSide = Math.max(src.width, src.height);
  // 원본보다 큰 스케일은 1.0으로 클램프되어 서로 수렴한다. 15% 이내로 붙은 스케일은
  // 사실상 같은 패스라 한 번만 돌린다.
  const effective: number[] = [];
  for (const s of scales.map((v) => Math.min(v, maxSide)).sort((a, b) => a - b)) {
    if (!effective.some((e) => Math.abs(e - s) / e < 0.15)) effective.push(s);
  }

  const all: TextBox[] = [];
  for (const s of effective) all.push(...(await detectAtScale(src, s)));

  all.sort((a, b) => b.score - a.score);
  const kept: TextBox[] = [];
  for (const box of all) {
    if (!kept.some((k) => boxIou(box.box, k.box) > 0.5)) kept.push(box);
  }
  return mergeIntoLines(kept);
}

/**
 * 같은 줄에 있는 인접 박스를 합친다. DBNet은 단어·글자 단위로 쪼개 내놓는 경우가 많은데,
 * 조각난 채로 인식하면 (1) 스트립 수가 늘어 느려지고 (2) 문맥이 끊겨 정확도가 떨어지며
 * (3) 검색어가 조각 경계에 걸리면 어느 격자에서도 완성되지 않는다.
 */
function mergeIntoLines(boxes: TextBox[]): TextBox[] {
  let current = [...boxes];

  for (let pass = 0; pass < 4; pass++) {
    const merged: TextBox[] = [];
    const used = new Set<number>();
    let changed = false;

    current.sort((a, b) => a.box[0] - b.box[0]);
    for (let i = 0; i < current.length; i++) {
      if (used.has(i)) continue;
      let [ax, ay, aw, ah] = current[i].box;
      let score = current[i].score;
      let count = 1;

      for (let j = i + 1; j < current.length; j++) {
        if (used.has(j)) continue;
        const [bx, by, bw, bh] = current[j].box;

        // 세로 범위가 대부분 겹치고 높이가 비슷하며 가로 간격이 좁아야 같은 줄로 본다.
        const overlap = Math.min(ay + ah, by + bh) - Math.max(ay, by);
        const minH = Math.min(ah, bh);
        if (overlap < minH * 0.6) continue;
        if (Math.max(ah, bh) / minH > 1.6) continue;
        const gap = bx - (ax + aw);
        if (gap > minH * 0.8 || gap < -minH) continue;

        const nx = Math.min(ax, bx);
        const ny = Math.min(ay, by);
        ax = nx;
        ay = ny;
        aw = Math.max(ax + aw, bx + bw) - nx;
        ah = Math.max(ay + ah, by + bh) - ny;
        score += current[j].score;
        count++;
        used.add(j);
        changed = true;
      }

      merged.push({ box: [ax, ay, aw, ah], score: score / count });
    }

    current = merged;
    if (!changed) break;
  }

  return current;
}

// ============================================================
// 인식 - CTC 확률 격자
// ============================================================

const REC_HEIGHT = 48;
/** 이보다 넓어지면 겹치며 분할한다. 압축하면 문자당 픽셀이 부족해 CTC가 문자를 분리하지 못한다. */
const REC_CHUNK_WIDTH = 1200;
/** 한 배치에 넣을 최대 스트립 수. 폭이 비슷한 것끼리 묶어 패딩 낭비를 줄인다. */
const REC_BATCH = 8;
/** 인식할 영역 수 상한. 다국어 rec 모델이 무거워 이 값이 사실상 최악 케이스 지연을 정한다. */
const MAX_REGIONS = 40;

/** [T, C] 확률 격자. */
interface Lattice {
  data: Float32Array;
  T: number;
  C: number;
}

/** 인식 대기 중인 48px 높이 스트립 (RGB raw). */
interface Strip {
  rgb: Buffer;
  width: number;
}

/**
 * 스트립을 폭 순으로 묶어 배치 추론한다. 한 장씩 돌리면 호출 오버헤드와 GEMM
 * 비효율로 영역 수에 비례해 급격히 느려진다. 패딩된 뒷부분은 실제 폭 비율만큼
 * 타임스텝을 잘라내 격자에서 제외한다.
 */
async function recognizeBatch(model: RecModel, strips: Strip[]): Promise<Lattice[]> {
  if (strips.length === 0) return [];

  const order = strips.map((s, i) => i).sort((a, b) => strips[a].width - strips[b].width);
  const lattices: Lattice[] = new Array(strips.length);

  for (let start = 0; start < order.length; start += REC_BATCH) {
    const group = order.slice(start, start + REC_BATCH);
    const maxW = Math.max(...group.map((i) => strips[i].width));
    const px = REC_HEIGHT * maxW;

    // 패딩은 정규화 후 0 (= 중간 회색). PaddleOCR도 정규화된 텐서를 0으로 채운다.
    const input = new Float32Array(group.length * 3 * px);
    group.forEach((idx, b) => {
      const { rgb, width } = strips[idx];
      const base = b * 3 * px;
      for (let c = 0; c < 3; c++) {
        for (let y = 0; y < REC_HEIGHT; y++) {
          for (let x = 0; x < width; x++) {
            input[base + c * px + y * maxW + x] = (rgb[(y * width + x) * 3 + c] / 255.0 - 0.5) / 0.5;
          }
        }
      }
    });

    const results = await model.session.run({
      [model.session.inputNames[0]]: new ort.Tensor("float32", input, [group.length, 3, REC_HEIGHT, maxW]),
    });
    const out = results[model.session.outputNames[0]];
    const dims = out.dims as number[];
    if (dims.length !== 3) continue;

    const [, T, C] = dims;
    const data = out.data as Float32Array;
    group.forEach((idx, b) => {
      const realT = Math.max(1, Math.ceil((T * strips[idx].width) / maxW));
      lattices[idx] = { data: data.subarray(b * T * C, b * T * C + realT * C), T: realT, C };
    });
  }

  return lattices.filter(Boolean);
}

/**
 * 세로쓰기 처리. 세로로 긴 영역을 그대로 높이 48로 리사이즈하면 폭이 한 자리 픽셀로
 * 뭉개져 정보가 전손된다. 縦書き는 글자가 대략 정사각 셀에 하나씩 쌓이므로,
 * 셀 단위로 잘라 가로로 이어 붙이면 글자를 세운 채 가로 텍스트로 만들 수 있다.
 */
function unstackVertical(crop: Buffer, w: number, h: number): { buf: Buffer; w: number; h: number } | null {
  const cells = Math.round(h / w);
  if (cells < 2 || cells > 40) return null;

  const cellH = Math.floor(h / cells);
  if (cellH < 8) return null;

  // 셀을 가로로 이어 붙인다. raw 버퍼 위 행 복사라 인코딩·디코딩이 없다.
  const outW = w * cells;
  const buf = Buffer.alloc(outW * cellH * 3, 255);
  for (let i = 0; i < cells; i++) {
    for (let y = 0; y < cellH; y++) {
      const srcOff = ((i * cellH + y) * w) * 3;
      const dstOff = (y * outW + i * w) * 3;
      crop.copy(buf, dstOff, srcOff, srcOff + w * 3);
    }
  }
  return { buf, w: outW, h: cellH };
}

/**
 * 한 텍스트 영역이 만들어내는 스트립들. 세로쓰기 분해본과 긴 줄 분할로 여러 개가 나온다.
 * 스트립은 모델과 무관한 이미지라 여러 언어 모델이 같은 스트립을 공유한다.
 */
async function buildStripsForRegion(
  src: RawSrc,
  box: [number, number, number, number],
  minStripWidth: number,
): Promise<Strip[]> {
  const left = Math.max(0, Math.min(Math.round(box[0]), src.width - 1));
  const top = Math.max(0, Math.min(Math.round(box[1]), src.height - 1));
  const width = Math.max(1, Math.min(Math.round(box[2]), src.width - left));
  const height = Math.max(1, Math.min(Math.round(box[3]), src.height - top));

  const crop = await fromRaw(src).extract({ left, top, width, height }).raw().toBuffer();

  const sources: { buf: Buffer; w: number; h: number }[] = [{ buf: crop, w: width, h: height }];

  // 세로로 길면 셀 분해본을 후보에 추가한다. 최고 점수를 취하므로 잘못된 가정이어도 손해가 없다.
  if (height > width * 1.5) {
    const unstacked = unstackVertical(crop, width, height);
    if (unstacked) sources.push(unstacked);
  }

  const strips: Strip[] = [];
  for (const s of sources) {
    const targetW = Math.max(8, Math.round(REC_HEIGHT * (s.w / s.h)));

    // 가장 짧은 검색어조차 담을 수 없을 만큼 좁은 스트립은 인식할 이유가 없다.
    // 48px 높이에서 문자당 12px 미만은 CJK로는 불가능한 밀도다.
    if (targetW < minStripWidth) continue;

    const resized = await sharp(s.buf, { raw: { width: s.w, height: s.h, channels: 3 } })
      .resize(targetW, REC_HEIGHT, { fit: "fill" })
      .raw()
      .toBuffer();

    // 너무 길면 겹치며 자른다. 겹침 덕에 경계에 걸친 검색어도 한쪽 청크에 온전히 들어간다.
    const chunks: [number, number][] = [];
    if (targetW <= REC_CHUNK_WIDTH) {
      chunks.push([0, targetW]);
    } else {
      const step = Math.floor(REC_CHUNK_WIDTH * 0.8);
      for (let x = 0; x < targetW; x += step) {
        const end = Math.min(x + REC_CHUNK_WIDTH, targetW);
        chunks.push([x, end]);
        if (end >= targetW) break;
      }
    }

    for (const [x0, x1] of chunks) {
      const cw = x1 - x0;
      if (cw < 8) continue;
      const strip = Buffer.alloc(REC_HEIGHT * cw * 3);
      for (let y = 0; y < REC_HEIGHT; y++) {
        resized.copy(strip, y * cw * 3, (y * targetW + x0) * 3, (y * targetW + x1) * 3);
      }
      strips.push({ rgb: strip, width: cw });
    }
  }

  return strips;
}

// ============================================================
// CTC lexicon 스코어링
// ============================================================

const NEG_INF = -1e30;

function logAdd(a: number, b: number): number {
  if (a === NEG_INF) return b;
  if (b === NEG_INF) return a;
  const hi = a > b ? a : b;
  const lo = a > b ? b : a;
  return hi + Math.log1p(Math.exp(lo - hi));
}

/**
 * 격자에서 target이 "어딘가에 나타날" 최대 확률을 CTC forward로 계산한다.
 *
 * 표준 CTC forced alignment는 격자 전체가 정확히 target이어야 하므로,
 * "제1화 大西亜玖璃 출연" 같은 줄에서 "大西亜玖璃"를 찾지 못한다. 그래서 상태 0을
 * "아직 시작 안 함"으로 두고 항상 확률 1로 유지해 앞쪽 임의 문자열을 허용하고,
 * 마지막 글자에 도달한 시점의 최대값을 취해 뒤쪽 임의 문자열도 허용한다.
 *
 * 반환값은 문자당 기하평균 확률 P^(1/L)이다. 길이가 다른 검색어끼리 비교 가능해진다.
 */
function ctcKeywordScore(lat: Lattice, classIds: number[][]): number {
  const L = classIds.length;
  if (L === 0) return 0;

  // 확장 시퀀스: [blank, c1, blank, c2, ..., cL, blank]
  // 각 실제 위치는 대체 문자 집합을 갖는다. 대표값(첫 원소)은 blank 전이 규칙 비교에 쓴다.
  const S = 2 * L + 1;
  const ext: number[][] = new Array(S);
  const primary = new Int32Array(S);
  for (let s = 0; s < S; s++) ext[s] = [0];
  for (let i = 0; i < L; i++) {
    ext[2 * i + 1] = classIds[i];
    primary[2 * i + 1] = classIds[i][0];
  }

  let alpha = new Float64Array(S).fill(NEG_INF);
  let next = new Float64Array(S);
  alpha[0] = 0; // 아직 시작 안 함 = 확률 1

  let best = NEG_INF;
  const { data, T, C } = lat;

  for (let t = 0; t < T; t++) {
    const base = t * C;
    next.fill(NEG_INF);
    next[0] = 0; // 앞쪽 임의 문자열 허용

    for (let s = 1; s < S; s++) {
      let acc = alpha[s];
      acc = logAdd(acc, alpha[s - 1]);
      if (s >= 2 && primary[s] !== 0 && primary[s] !== primary[s - 2]) acc = logAdd(acc, alpha[s - 2]);
      if (acc === NEG_INF) continue;

      // 대체 문자 중 가장 높은 확률을 쓴다. 인식기가 ヶ를 久로 읽어도 매칭이 살아남는다.
      let p = 0;
      for (const id of ext[s]) {
        const v = data[base + id];
        if (v > p) p = v;
      }
      next[s] = acc + Math.log(p > 1e-12 ? p : 1e-12);
    }

    const tmp = alpha;
    alpha = next;
    next = tmp;

    // 마지막 실제 글자에 도달했으면 완성. 뒤에 무엇이 오든 상관없다.
    if (alpha[S - 2] > best) best = alpha[S - 2];
    if (alpha[S - 1] > best) best = alpha[S - 1];
  }

  if (best === NEG_INF) return 0;
  return Math.exp(best / L);
}

/**
 * 검색어를 위치별 클래스 인덱스 집합으로 바꾼다. 어느 위치든 사전에 표현 가능한 문자가
 * 하나도 없으면 이 모델로는 그 검색어를 채점할 수 없다.
 */
function toClassIdSets(text: string, model: RecModel): number[][] | null {
  const sets: number[][] = [];
  for (const ch of text) {
    const ids: number[] = [];
    const primary = model.charIndex.get(ch);
    if (primary !== undefined) ids.push(primary);
    for (const alt of alternativesFor(ch)) {
      const id = model.charIndex.get(alt);
      if (id !== undefined && !ids.includes(id)) ids.push(id);
    }
    if (ids.length === 0) return null;
    sets.push(ids);
  }
  return sets.length > 0 ? sets : null;
}

// ============================================================
// 파이프라인
// ============================================================

export interface OcrPartScore {
  text: string;
  score: number;
}

export interface OcrResult {
  /** 일치한 리스트의 원본 줄. 하나라도 있으면 성공. */
  found: string[];
  /** 리스트별 파트 점수 (임계값 조정용). */
  detail: { list: string; matched: boolean; parts: OcrPartScore[] }[];
  regions: number;
}

export async function performOCR(
  imageBuffer: Buffer,
  searchLists: SearchList[],
  opts: { scoreThreshold: number; detScales: number[] },
): Promise<OcrResult> {
  const empty: OcrResult = { found: [], detail: [], regions: 0 };

  if (!isOcrReady()) {
    console.log(`[ocr] SKIP - det: ${!!detSession}, rec: ${recModels.size}`);
    return empty;
  }
  if (searchLists.length === 0) {
    console.log(`[ocr] SKIP - searchLists is empty (searchStrings.tsv를 확인하라)`);
    return empty;
  }

  // 소스를 한 번만 디코딩한다. 이후 모든 크롭·리사이즈가 이 raw 버퍼를 공유한다.
  const decoded = await sharp(imageBuffer).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const src: RawSrc = { buf: decoded.data, width: decoded.info.width, height: decoded.info.height };

  const tDet = Date.now();
  const boxes = await detectTextRegions(src, opts.detScales);
  const detMs = Date.now() - tDet;
  if (boxes.length === 0) {
    console.log(`[ocr] no text regions detected - scales: ${opts.detScales.join(",")}, detMs: ${detMs}`);
    return empty;
  }

  // 스트립은 모델과 무관하므로 한 번만 만들어 모든 언어 모델이 공유한다.
  const shortestKeyword = Math.min(...searchLists.flatMap((l) => l.parts.map((p) => [...p].length)));
  const minStripWidth = Math.max(16, 12 * shortestKeyword);

  // 인식이 전체 시간의 대부분이라 최악 케이스에 상한을 둔다. 검출 점수가 높은
  // 영역부터 처리하므로 잘리는 것은 신뢰도가 낮은 영역이다.
  const tStrip = Date.now();
  const ranked = [...boxes].sort((a, b) => b.score - a.score).slice(0, MAX_REGIONS);
  const strips: Strip[] = [];
  for (const tb of ranked) strips.push(...(await buildStripsForRegion(src, tb.box, minStripWidth)));
  const stripMs = Date.now() - tStrip;

  // 검색어를 표현할 수 있는 모델은 전부 돌리고 점수로 승부를 가린다.
  //
  // 표기만 보고 모델을 하나 고르면 안 된다. 한자는 일본어·중국어가 공유하므로 문자 종류로는
  // 어느 모델이 잘 읽을지 알 수 없고, 사전에 있다는 것(표현 가능)과 잘 읽는다는 것(학습 언어)은
  // 다른 문제다. 실측: 일본어 이름을 중국어 모델로 읽히면 간체자로 끌린다(亜→亚, 謝→谢).
  //
  // 다만 무조건 전부 돌리지는 않는다. toClassIdSets가 null이면 그 모델의 점수는 반드시 0이라
  // 인식을 돌려도 결과가 정해져 있다. 인식이 전체 시간의 대부분이므로 그런 모델은 건너뛴다.
  // 예: ko 사전은 한자·가나가 0자라 `大西亜玖璃`를 한 글자도 표현하지 못한다.
  const setsByLang = new Map<RecLang, Map<string, number[][]>>();
  const unrepresentable = new Set<string>();
  for (const list of searchLists) {
    for (const part of list.parts) {
      let anyLang = false;
      for (const [lang, model] of recModels) {
        const sets = toClassIdSets(part, model);
        if (!sets) continue;
        anyLang = true;
        let perPart = setsByLang.get(lang);
        if (!perPart) setsByLang.set(lang, (perPart = new Map()));
        perPart.set(part, sets);
      }
      if (!anyLang) unrepresentable.add(part);
    }
  }
  for (const part of unrepresentable) {
    console.warn(`Search part not representable - part: "${part}" (어느 사전에도 없는 문자, 매칭 불가)`);
  }

  const tRec = Date.now();
  const latticesByLang = new Map<RecLang, Lattice[]>();
  for (const lang of setsByLang.keys()) {
    latticesByLang.set(lang, await recognizeBatch(recModels.get(lang)!, strips));
  }
  const recMs = Date.now() - tRec;

  // 파트 하나의 점수 = 표현 가능한 모든 모델 × 모든 격자 × 모든 표기 변형 중 최대값
  const scorePart = (part: string): number => {
    let best = 0;
    for (const [lang, perPart] of setsByLang) {
      const sets = perPart.get(part);
      const lattices = latticesByLang.get(lang);
      if (!sets || !lattices) continue;
      for (const lat of lattices) {
        const s = ctcKeywordScore(lat, sets);
        if (s > best) best = s;
      }
    }
    return best;
  };

  const detail: OcrResult["detail"] = [];
  const found: string[] = [];
  for (const list of searchLists) {
    const parts = list.parts.map((text) => ({ text, score: scorePart(text) }));
    const matched = parts.every((p) => p.score >= opts.scoreThreshold);
    detail.push({ list: list.raw, matched, parts });
    if (matched) found.push(list.raw);
  }

  console.log(
    `[ocr] regions: ${boxes.length}, strips: ${strips.length}, lattices: ${[...latticesByLang].map(([l, v]) => `${l}=${v.length}`).join(",")}, matched: ${found.length}/${searchLists.length}, detMs: ${detMs}, stripMs: ${stripMs}, recMs: ${recMs}`,
  );
  for (const d of detail) {
    console.log(`[ocr]   list="${d.list.replace(/\t/g, "\\t")}" matched=${d.matched} parts=${d.parts.map((p) => `${p.text}:${p.score.toFixed(3)}`).join(" ")}`);
  }

  return { found, detail, regions: boxes.length };
}
