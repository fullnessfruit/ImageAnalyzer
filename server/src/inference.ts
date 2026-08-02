/**
 * 모든 ONNX 추론.
 *
 * 세션은 모듈 레벨 싱글톤으로 시작 시 한 번만 로딩한다.
 *
 * 모델별 역할:
 *  - face-det (SCRFD det_10g) + arcface  : 실사 인물. 랜드마크 정합이 필수다.
 *  - anime-face-det / anime-person-det   : 애니 영역 검출 (YOLOv8, 학습 불필요)
 *  - ccip                                : 캐릭터 신원. 화풍이 달라도 같은 캐릭터를 모은다.
 *  - wd-tagger                           : 캐릭터 이름(제로샷) + 의류 태그(의상 표현)
 */

import * as ort from "onnxruntime-node";
import sharp from "sharp";
import fs from "fs";
import path from "path";
import { getModelPath, ModelName } from "./model-downloader";

let arcfaceSession: ort.InferenceSession | null = null;
let faceDetSession: ort.InferenceSession | null = null;
let animeFaceDetSession: ort.InferenceSession | null = null;
let animePersonDetSession: ort.InferenceSession | null = null;
let ccipSession: ort.InferenceSession | null = null;
let wdTaggerSession: ort.InferenceSession | null = null;

/** CCIP 입력 해상도. 모델 메타데이터에서 읽고, 동적이면 384를 쓴다. */
let ccipSize = 384;
/** WD-Tagger 입력 해상도. v3 계열은 448. */
let wdSize = 448;

export interface WdTag {
  name: string;
  category: number; // 0=general, 4=character, 9=rating
  count: number;
  /** log(N/count). 희귀 태그일수록 의상 식별력이 높다. */
  idf: number;
}
let wdTags: WdTag[] = [];
/** wdTags 중 의류·장신구에 해당하는 인덱스. 의상 벡터는 이 부분집합만 쓴다. */
let clothingTagIndices: number[] = [];

// ============================================================
// 로딩
// ============================================================

async function tryLoad(modelsDir: string, name: ModelName, label: string): Promise<ort.InferenceSession | null> {
  const p = getModelPath(modelsDir, name);
  if (!fs.existsSync(p)) {
    console.warn(`⚠️  Model missing - name: ${label}, path: ${p} (해당 기능 비활성)`);
    return null;
  }
  try {
    const session = await ort.InferenceSession.create(p, { intraOpNumThreads: 4, graphOptimizationLevel: "all" });
    console.log(`✅ Model loaded - name: ${label}, inputs: ${session.inputNames.join(",")}, outputs: ${session.outputNames.length}`);
    return session;
  } catch (e: any) {
    console.error(`❌ Model load failed - name: ${label}, path: ${p}, error: ${e.message}`);
    return null;
  }
}

export async function loadModels(modelsDir: string): Promise<void> {
  faceDetSession = await tryLoad(modelsDir, "face-det", "face-det(SCRFD)");
  arcfaceSession = await tryLoad(modelsDir, "arcface", "arcface");
  animeFaceDetSession = await tryLoad(modelsDir, "anime-face-det", "anime-face-det");
  animePersonDetSession = await tryLoad(modelsDir, "anime-person-det", "anime-person-det");
  ccipSession = await tryLoad(modelsDir, "ccip", "ccip");
  wdTaggerSession = await tryLoad(modelsDir, "wd-tagger", "wd-tagger");

  if (ccipSession) {
    const dims = (ccipSession.inputMetadata as any)?.[ccipSession.inputNames[0]]?.shape;
    const h = Array.isArray(dims) ? dims[2] : undefined;
    if (typeof h === "number" && h > 0) ccipSize = h;
    console.log(`[ccip] input size: ${ccipSize}`);
  }

  if (wdTaggerSession) {
    const dims = (wdTaggerSession.inputMetadata as any)?.[wdTaggerSession.inputNames[0]]?.shape;
    const h = Array.isArray(dims) ? dims[1] : undefined;
    if (typeof h === "number" && h > 0) wdSize = h;
    loadWdTags(modelsDir);
    console.log(`[wd-tagger] input size: ${wdSize}, tags: ${wdTags.length}, clothing tags: ${clothingTagIndices.length}`);
  }
}

/**
 * 의류·장신구 태그 판별. 태그 이름에 아래 패턴이 있으면 의상 구성요소로 본다.
 * 머리색·눈색·체형 같은 착용자 고유 속성은 의도적으로 제외한다 — 그래야 다른 캐릭터가
 * 같은 옷을 입어도 같은 의상으로 매칭된다.
 */
const CLOTHING_PATTERN =
  /(dress|skirt|shirt|blouse|jacket|coat|uniform|sleeve|collar|ribbon|bowtie|necktie|_tie$|^tie$|hat|cap$|beret|helmet|hood|glove|thighhigh|kneehigh|sock|stocking|pantyhose|legwear|boot|shoe|sandal|heel|scarf|apron|cape|cloak|armor|armour|kimono|yukata|hakama|swimsuit|bikini|leotard|belt|buckle|corset|vest|sweater|hoodie|shorts|pants|trouser|jean|overall|cardigan|veil|headband|hairband|hair_ornament|hair_bow|hair_flower|hairclip|earring|necklace|pendant|bracelet|choker|brooch|badge|epaulette|frill|lace|ruffle|button|zipper|pocket|robe|tunic|toga|sash|obi$|_obi|garter|suspender|bodysuit|jumpsuit|romper|petticoat|bloomers|panties|bra$|lingerie|underwear|mask|goggles|glasses|eyepatch|headphones|headgear|crown|tiara|halo|wing_collar|shoulder_armor|gauntlet|pauldron|bandage|bandaid|scrunchie|hairpin|barrette)/i;

function loadWdTags(modelsDir: string): void {
  const csvPath = getModelPath(modelsDir, "wd-tags");
  if (!fs.existsSync(csvPath)) {
    console.warn(`⚠️  WD tag list missing - path: ${csvPath}`);
    return;
  }
  const lines = fs.readFileSync(csvPath, "utf-8").split("\n");
  const header = lines[0].split(",").map((s) => s.trim());
  const iName = header.indexOf("name");
  const iCat = header.indexOf("category");
  const iCount = header.indexOf("count");

  const parsed: WdTag[] = [];
  let totalCount = 0;
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    if (cols.length <= iName) continue;
    const count = parseInt(cols[iCount] ?? "0", 10) || 1;
    parsed.push({ name: cols[iName], category: parseInt(cols[iCat] ?? "0", 10) || 0, count, idf: 0 });
    totalCount = Math.max(totalCount, count);
  }
  // IDF 기준값은 최다 빈도 태그. 희귀 태그일수록 가중치가 커진다.
  for (const t of parsed) t.idf = Math.log(totalCount / t.count) + 1;

  wdTags = parsed;
  clothingTagIndices = [];
  for (let i = 0; i < wdTags.length; i++) {
    if (wdTags[i].category === 0 && CLOTHING_PATTERN.test(wdTags[i].name)) clothingTagIndices.push(i);
  }
}

export function hasWdTagger(): boolean {
  return wdTaggerSession !== null && wdTags.length > 0;
}
export function hasCcip(): boolean {
  return ccipSession !== null;
}
export function hasArcFace(): boolean {
  return arcfaceSession !== null && faceDetSession !== null;
}

// ============================================================
// 이미지 유틸
// ============================================================

export interface RawImage {
  data: Buffer; // RGB, 3 bytes/px
  width: number;
  height: number;
}

/** 요청당 한 번만 디코딩해서 얼굴 정합·크롭에 재사용한다. */
export async function loadRaw(imageBuffer: Buffer): Promise<RawImage> {
  const { data, info } = await sharp(imageBuffer)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

export type Box = [number, number, number, number]; // x, y, w, h

export interface Detection {
  box: Box;
  confidence: number;
  classId: number;
  /** SCRFD만 제공. [x,y] × 5 — 좌눈, 우눈, 코, 좌입꼬리, 우입꼬리. */
  landmarks?: number[];
}

export async function cropRegion(imageBuffer: Buffer, box: Box): Promise<Buffer> {
  const meta = await sharp(imageBuffer).metadata();
  const imgW = meta.width!;
  const imgH = meta.height!;
  const left = Math.max(0, Math.min(Math.round(box[0]), imgW - 1));
  const top = Math.max(0, Math.min(Math.round(box[1]), imgH - 1));
  const width = Math.max(1, Math.min(Math.round(box[2]), imgW - left));
  const height = Math.max(1, Math.min(Math.round(box[3]), imgH - top));
  return sharp(imageBuffer).extract({ left, top, width, height }).toBuffer();
}

/**
 * box 영역을 크롭하고 그 안에 들어오는 maskBoxes를 회색으로 덮는다.
 * 의상 인식에서 머리·얼굴을 지우는 데 쓴다 — 착용자 신원 정보를 제거해야
 * 다른 캐릭터가 같은 옷을 입어도 같은 의상으로 매칭된다.
 * 머리카락까지 덮도록 얼굴 위쪽에 높이의 60%를 더 확장한다.
 */
export async function cropWithMask(imageBuffer: Buffer, box: Box, maskBoxes: Box[]): Promise<Buffer> {
  const [bx, by, bw, bh] = box.map(Math.round) as Box;
  const cropped = await cropRegion(imageBuffer, box);

  const overlays: sharp.OverlayOptions[] = [];
  for (const m of maskBoxes) {
    const padTop = Math.round(m[3] * 0.6);
    const left = Math.round(m[0] - bx);
    const top = Math.round(m[1] - by - padTop);
    const right = left + Math.round(m[2]);
    const bottom = top + Math.round(m[3]) + padTop;

    // 크롭 영역과 교차하는 부분만 남긴다.
    const cl = Math.max(0, left);
    const ct = Math.max(0, top);
    const cr = Math.min(bw, right);
    const cb = Math.min(bh, bottom);
    if (cr <= cl || cb <= ct) continue;

    overlays.push({
      input: {
        create: { width: cr - cl, height: cb - ct, channels: 3, background: { r: 128, g: 128, b: 128 } },
      },
      left: cl,
      top: ct,
    });
  }

  if (overlays.length === 0) return cropped;
  return sharp(cropped).composite(overlays).toBuffer();
}

// ============================================================
// 전처리
// ============================================================

/**
 * 종횡비를 유지하며 targetSize 안에 넣는다. YOLOv8은 가운데 정렬 + 114 패딩,
 * SCRFD는 좌상단 정렬 + 0 패딩(InsightFace 구현과 동일)이라 두 방식을 모두 지원한다.
 */
async function letterbox(
  imageBuffer: Buffer,
  targetSize: number,
  opts: { center: boolean; pad: number },
): Promise<{ data: Buffer; ratio: number; padX: number; padY: number }> {
  const meta = await sharp(imageBuffer).metadata();
  const ratio = Math.min(targetSize / meta.width!, targetSize / meta.height!);
  const newW = Math.round(meta.width! * ratio);
  const newH = Math.round(meta.height! * ratio);
  const padX = opts.center ? Math.floor((targetSize - newW) / 2) : 0;
  const padY = opts.center ? Math.floor((targetSize - newH) / 2) : 0;

  const { data } = await sharp(imageBuffer)
    .resize(newW, newH, { fit: "fill" })
    .extend({
      top: padY,
      bottom: targetSize - newH - padY,
      left: padX,
      right: targetSize - newW - padX,
      background: { r: opts.pad, g: opts.pad, b: opts.pad },
    })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  return { data, ratio, padX, padY };
}

/** HWC uint8 → NCHW float32. */
function toNCHW(data: Buffer, size: number, scale: number, mean: number[], std: number[]): Float32Array {
  const px = size * size;
  const out = new Float32Array(3 * px);
  for (let c = 0; c < 3; c++) {
    const base = c * px;
    for (let i = 0; i < px; i++) {
      out[base + i] = (data[i * 3 + c] * scale - mean[c]) / std[c];
    }
  }
  return out;
}

// ============================================================
// 실사 얼굴 검출 (SCRFD det_10g) — 랜드마크 포함
// ============================================================

const SCRFD_SIZE = 640;
const SCRFD_STRIDES = [8, 16, 32];
const SCRFD_ANCHORS_PER_CELL = 2;

export async function detectFaces(imageBuffer: Buffer, confThreshold = 0.5): Promise<Detection[]> {
  if (!faceDetSession) return [];

  // InsightFace는 리사이즈한 이미지를 좌상단에 놓고 나머지를 0으로 채운다.
  const { data, ratio } = await letterbox(imageBuffer, SCRFD_SIZE, { center: false, pad: 0 });
  const input = toNCHW(data, SCRFD_SIZE, 1, [127.5, 127.5, 127.5], [128, 128, 128]);
  const results = await faceDetSession.run({
    [faceDetSession.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, SCRFD_SIZE, SCRFD_SIZE]),
  });

  // 출력 9개: 3 FPN 레벨 × (score [N,1], bbox [N,4], kps [N,10]).
  // 이름 순서에 의존하지 않도록 열 수로 그룹핑하고 앵커 수 내림차순(stride 8→16→32)으로 정렬한다.
  const outputs = faceDetSession.outputNames.map((n) => results[n]);
  const byCols = (c: number) =>
    outputs.filter((o) => o.dims[1] === c).sort((a, b) => (b.dims[0] as number) - (a.dims[0] as number));
  const scoreOut = byCols(1);
  const bboxOut = byCols(4);
  const kpsOut = byCols(10);

  if (scoreOut.length !== 3 || bboxOut.length !== 3) {
    console.error(`❌ SCRFD output shape unexpected - scores: ${scoreOut.length}, bboxes: ${bboxOut.length}, kps: ${kpsOut.length}`);
    return [];
  }

  const detections: Detection[] = [];
  const inv = 1 / ratio;

  for (let si = 0; si < 3; si++) {
    const stride = SCRFD_STRIDES[si];
    const scores = scoreOut[si].data as Float32Array;
    const bboxes = bboxOut[si].data as Float32Array;
    const kps = kpsOut.length === 3 ? (kpsOut[si].data as Float32Array) : null;
    const fm = SCRFD_SIZE / stride;

    let idx = 0;
    for (let row = 0; row < fm; row++) {
      for (let col = 0; col < fm; col++) {
        const ax = col * stride;
        const ay = row * stride;
        for (let a = 0; a < SCRFD_ANCHORS_PER_CELL; a++) {
          const score = scores[idx];
          if (score >= confThreshold) {
            const x1 = (ax - bboxes[idx * 4 + 0] * stride) * inv;
            const y1 = (ay - bboxes[idx * 4 + 1] * stride) * inv;
            const x2 = (ax + bboxes[idx * 4 + 2] * stride) * inv;
            const y2 = (ay + bboxes[idx * 4 + 3] * stride) * inv;

            let landmarks: number[] | undefined;
            if (kps) {
              landmarks = [];
              for (let k = 0; k < 5; k++) {
                landmarks.push((ax + kps[idx * 10 + k * 2] * stride) * inv);
                landmarks.push((ay + kps[idx * 10 + k * 2 + 1] * stride) * inv);
              }
            }

            detections.push({
              box: [x1, y1, x2 - x1, y2 - y1],
              confidence: score,
              classId: 0,
              landmarks,
            });
          }
          idx++;
        }
      }
    }
  }

  return nms(detections, 0.4);
}

// ============================================================
// 얼굴 정합 (5점 유사변환)
// ============================================================

/** ArcFace 정규 템플릿. 이 배치로 워프된 얼굴로만 학습되었다. */
const ARCFACE_TEMPLATE = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];
const ARCFACE_SIZE = 112;

/**
 * 최소제곱 유사변환 src→dst 추정.
 *   x' = a·x − b·y + tx
 *   y' = b·x + a·y + ty
 * 회전·스케일·평행이동 4자유도라 정규방정식이 닫힌 형태로 풀린다(a와 b가 분리된다).
 */
function estimateSimilarity(src: number[][], dst: number[][]): { a: number; b: number; tx: number; ty: number } {
  const n = src.length;
  let mx = 0, my = 0, mx2 = 0, my2 = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i][0];
    my += src[i][1];
    mx2 += dst[i][0];
    my2 += dst[i][1];
  }
  mx /= n; my /= n; mx2 /= n; my2 /= n;

  let sa = 0, sb = 0, d = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - mx;
    const y = src[i][1] - my;
    const u = dst[i][0] - mx2;
    const v = dst[i][1] - my2;
    sa += x * u + y * v;
    sb += x * v - y * u;
    d += x * x + y * y;
  }
  if (d === 0) return { a: 1, b: 0, tx: 0, ty: 0 };

  const a = sa / d;
  const b = sb / d;
  return { a, b, tx: mx2 - a * mx + b * my, ty: my2 - b * mx - a * my };
}

/**
 * 랜드마크로 얼굴을 ArcFace 템플릿에 워프한다. 출력 픽셀에서 원본을 역방향으로
 * 이중선형 샘플링하므로 회전·스케일이 정확히 반영된다.
 */
export function alignFace(raw: RawImage, landmarks: number[]): Buffer {
  const src: number[][] = [];
  for (let k = 0; k < 5; k++) src.push([landmarks[k * 2], landmarks[k * 2 + 1]]);
  const { a, b, tx, ty } = estimateSimilarity(src, ARCFACE_TEMPLATE);

  const det = a * a + b * b;
  const out = Buffer.alloc(ARCFACE_SIZE * ARCFACE_SIZE * 3);
  if (det === 0) return out;

  for (let v = 0; v < ARCFACE_SIZE; v++) {
    for (let u = 0; u < ARCFACE_SIZE; u++) {
      // 역변환: (u,v) → 원본 (x,y)
      const du = u - tx;
      const dv = v - ty;
      const x = (a * du + b * dv) / det;
      const y = (-b * du + a * dv) / det;

      const o = (v * ARCFACE_SIZE + u) * 3;
      if (x < 0 || y < 0 || x >= raw.width - 1 || y >= raw.height - 1) continue;

      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const fx = x - x0;
      const fy = y - y0;
      const i00 = (y0 * raw.width + x0) * 3;
      const i10 = i00 + 3;
      const i01 = i00 + raw.width * 3;
      const i11 = i01 + 3;

      for (let c = 0; c < 3; c++) {
        const top = raw.data[i00 + c] * (1 - fx) + raw.data[i10 + c] * fx;
        const bot = raw.data[i01 + c] * (1 - fx) + raw.data[i11 + c] * fx;
        out[o + c] = top * (1 - fy) + bot * fy;
      }
    }
  }
  return out;
}

/**
 * 정합된 얼굴에서 ArcFace 임베딩 추출. 원본과 좌우반전을 각각 임베딩해
 * L2 정규화 후 평균한다(flip TTA) — 얼굴인식의 표준 기법으로 포즈 편향을 줄인다.
 */
export async function extractFaceEmbedding(aligned: Buffer): Promise<Float32Array | null> {
  if (!arcfaceSession) return null;

  const px = ARCFACE_SIZE * ARCFACE_SIZE;
  const makeTensor = (flip: boolean): ort.Tensor => {
    const arr = new Float32Array(3 * px);
    for (let y = 0; y < ARCFACE_SIZE; y++) {
      for (let x = 0; x < ARCFACE_SIZE; x++) {
        const sx = flip ? ARCFACE_SIZE - 1 - x : x;
        const si = (y * ARCFACE_SIZE + sx) * 3;
        const di = y * ARCFACE_SIZE + x;
        for (let c = 0; c < 3; c++) arr[c * px + di] = (aligned[si + c] - 127.5) / 127.5;
      }
    }
    return new ort.Tensor("float32", arr, [1, 3, ARCFACE_SIZE, ARCFACE_SIZE]);
  };

  const name = arcfaceSession.inputNames[0];
  const [r1, r2] = await Promise.all([
    arcfaceSession.run({ [name]: makeTensor(false) }),
    arcfaceSession.run({ [name]: makeTensor(true) }),
  ]);

  const e1 = r1[arcfaceSession.outputNames[0]].data as Float32Array;
  const e2 = r2[arcfaceSession.outputNames[0]].data as Float32Array;

  const norm = (v: Float32Array) => {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    s = Math.sqrt(s) || 1;
    const o = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) o[i] = v[i] / s;
    return o;
  };

  const n1 = norm(e1);
  const n2 = norm(e2);
  const out = new Float32Array(n1.length);
  for (let i = 0; i < n1.length; i++) out[i] = (n1[i] + n2[i]) / 2;
  return out;
}

/**
 * 품질 게이트. 너무 작거나 포즈가 극단적인 얼굴은 임베딩이 신뢰할 수 없어
 * 갤러리와 질의 양쪽을 오염시킨다. 매칭을 시도하지 않는 편이 낫다.
 */
export function faceQualityOk(det: Detection, minSize = 40): boolean {
  if (det.box[2] < minSize || det.box[3] < minSize) return false;
  if (!det.landmarks) return true;

  const [lx, ly, rx, ry, nx, ny] = det.landmarks;
  const eyeDist = Math.hypot(rx - lx, ry - ly);
  if (eyeDist < minSize * 0.25) return false;

  // 코가 두 눈 중점에서 눈 간격의 절반 넘게 벗어나면 심한 측면으로 본다.
  const midX = (lx + rx) / 2;
  const midY = (ly + ry) / 2;
  return Math.hypot(nx - midX, ny - midY) <= eyeDist * 0.9;
}

// ============================================================
// 애니 검출 (YOLOv8)
// ============================================================

const YOLO_SIZE = 640;

async function runYolo(
  session: ort.InferenceSession,
  imageBuffer: Buffer,
  confThreshold: number,
): Promise<Detection[]> {
  const { data, ratio, padX, padY } = await letterbox(imageBuffer, YOLO_SIZE, { center: true, pad: 114 });
  const input = toNCHW(data, YOLO_SIZE, 1 / 255, [0, 0, 0], [1, 1, 1]);
  const results = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, YOLO_SIZE, YOLO_SIZE]),
  });

  const output = results[session.outputNames[0]];
  const d = output.data as Float32Array;
  const dims = output.dims as number[];
  const inv = 1 / ratio;

  // YOLOv8 export는 [1, 4+nc, N]이 표준이지만 [1, N, 4+nc]로 나오는 변형도 있다.
  // 필드 수는 항상 박스 수보다 훨씬 작으므로 작은 쪽을 필드 축으로 본다.
  const transposed = dims[1] > dims[2];
  const numFields = transposed ? dims[2] : dims[1];
  const numBoxes = transposed ? dims[1] : dims[2];
  const at = (field: number, box: number) => (transposed ? d[box * numFields + field] : d[field * numBoxes + box]);

  const detections: Detection[] = [];
  for (let i = 0; i < numBoxes; i++) {
    let maxConf = 0;
    let maxClass = 0;
    for (let c = 4; c < numFields; c++) {
      const conf = at(c, i);
      if (conf > maxConf) {
        maxConf = conf;
        maxClass = c - 4;
      }
    }
    if (maxConf < confThreshold) continue;

    const cx = at(0, i);
    const cy = at(1, i);
    const w = at(2, i);
    const h = at(3, i);
    detections.push({
      box: [(cx - w / 2 - padX) * inv, (cy - h / 2 - padY) * inv, w * inv, h * inv],
      confidence: maxConf,
      classId: maxClass,
    });
  }

  return nms(detections, 0.45);
}

export async function detectAnimeFaces(imageBuffer: Buffer, conf = 0.35): Promise<Detection[]> {
  return animeFaceDetSession ? runYolo(animeFaceDetSession, imageBuffer, conf) : [];
}

export async function detectAnimePersons(imageBuffer: Buffer, conf = 0.35): Promise<Detection[]> {
  return animePersonDetSession ? runYolo(animePersonDetSession, imageBuffer, conf) : [];
}

// ============================================================
// CCIP — 캐릭터 신원 임베딩
// ============================================================

/**
 * 같은 캐릭터를 서로 다른 작가·화풍에 걸쳐 가깝게 두도록 학습된 공간.
 * CLIP처럼 화풍·구도로 뭉치지 않으므로 새 일러스트·동인 이미지에도 신원이 유지된다.
 */
export async function extractCcipEmbedding(imageBuffer: Buffer): Promise<Float32Array | null> {
  if (!ccipSession) return null;
  const { data } = await sharp(imageBuffer)
    .resize(ccipSize, ccipSize, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const input = toNCHW(data, ccipSize, 1 / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
  const results = await ccipSession.run({
    [ccipSession.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, ccipSize, ccipSize]),
  });
  return new Float32Array(results[ccipSession.outputNames[0]].data as Float32Array);
}

// ============================================================
// WD-Tagger — 캐릭터 이름(제로샷) + 의상 태그 벡터
// ============================================================

export interface TaggerResult {
  /** 임계값을 넘은 캐릭터 태그. 갤러리 등록 없이 이름을 얻는다. */
  characters: { name: string; prob: number }[];
  /** 의류 태그만 골라 IDF 가중한 벡터. 화풍·착용자에 불변인 의상 표현. */
  costumeVector: Float32Array;
  /** 상위 의류 태그 (디버깅·설명용). */
  topClothing: { name: string; prob: number }[];
}

/** 태그별 원시 확률. 임계값 조정과 진단에 쓴다. */
export async function taggerProbs(imageBuffer: Buffer): Promise<{ probs: Float32Array; tags: WdTag[] } | null> {
  if (!wdTaggerSession || wdTags.length === 0) return null;
  const probs = await runTaggerRaw(imageBuffer);
  return probs ? { probs, tags: wdTags } : null;
}

async function runTaggerRaw(imageBuffer: Buffer): Promise<Float32Array | null> {
  if (!wdTaggerSession) return null;

  // SmilingWolf v3 계열: 흰 배경으로 정사각 패딩 → 448 리사이즈 → BGR, 0~255 그대로, NHWC.
  // sharp는 체이닝 순서와 무관하게 resize를 extend보다 먼저 적용하므로, 패딩을 먼저
  // 확정하려면 별도 파이프라인으로 나눠야 한다. 한 번에 체이닝하면 종횡비가 뭉개진다.
  const meta = await sharp(imageBuffer).metadata();
  const side = Math.max(meta.width!, meta.height!);
  const squared = await sharp(imageBuffer)
    .removeAlpha()
    .extend({
      top: Math.floor((side - meta.height!) / 2),
      bottom: Math.ceil((side - meta.height!) / 2),
      left: Math.floor((side - meta.width!) / 2),
      right: Math.ceil((side - meta.width!) / 2),
      background: { r: 255, g: 255, b: 255 },
    })
    .png()
    .toBuffer();

  const { data } = await sharp(squared)
    .resize(wdSize, wdSize, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const px = wdSize * wdSize;
  const input = new Float32Array(px * 3);
  for (let i = 0; i < px; i++) {
    input[i * 3 + 0] = data[i * 3 + 2]; // B
    input[i * 3 + 1] = data[i * 3 + 1]; // G
    input[i * 3 + 2] = data[i * 3 + 0]; // R
  }

  const results = await wdTaggerSession.run({
    [wdTaggerSession.inputNames[0]]: new ort.Tensor("float32", input, [1, wdSize, wdSize, 3]),
  });
  return results[wdTaggerSession.outputNames[0]].data as Float32Array;
}

export async function runTagger(imageBuffer: Buffer, charThreshold = 0.6): Promise<TaggerResult | null> {
  if (!wdTaggerSession || wdTags.length === 0) return null;
  const probs = await runTaggerRaw(imageBuffer);
  if (!probs) return null;

  const characters: { name: string; prob: number }[] = [];
  for (let i = 0; i < wdTags.length && i < probs.length; i++) {
    if (wdTags[i].category === 4 && probs[i] >= charThreshold) {
      characters.push({ name: wdTags[i].name, prob: probs[i] });
    }
  }
  characters.sort((a, b) => b.prob - a.prob);

  const costumeVector = new Float32Array(clothingTagIndices.length);
  const topClothing: { name: string; prob: number }[] = [];
  for (let k = 0; k < clothingTagIndices.length; k++) {
    const ti = clothingTagIndices[k];
    const p = ti < probs.length ? probs[ti] : 0;
    costumeVector[k] = p * wdTags[ti].idf;
    if (p >= 0.35) topClothing.push({ name: wdTags[ti].name, prob: p });
  }
  topClothing.sort((a, b) => b.prob - a.prob);

  return { characters, costumeVector, topClothing: topClothing.slice(0, 15) };
}

// ============================================================
// NMS
// ============================================================

function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

function nms(detections: Detection[], iouThreshold: number): Detection[] {
  detections.sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  for (const det of detections) {
    if (!kept.some((k) => iou(det.box, k.box) > iouThreshold)) kept.push(det);
  }
  return kept;
}

/** box 안에 중심이 들어오는 검출들을 고른다. 사람 영역 안의 얼굴 찾기 등에 쓴다. */
export function detectionsInside(box: Box, candidates: Detection[]): Detection[] {
  return candidates.filter((c) => {
    const cx = c.box[0] + c.box[2] / 2;
    const cy = c.box[1] + c.box[3] / 2;
    return cx >= box[0] && cx <= box[0] + box[2] && cy >= box[1] && cy <= box[1] + box[3];
  });
}
