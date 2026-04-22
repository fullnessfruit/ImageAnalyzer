/**
 * OCR Pipeline: PaddleOCR (detection + recognition) + manga-ocr (Japanese fallback)
 *
 * Flow:
 *   1. PaddleOCR DBNet detects text regions
 *   2. PaddleOCR CRNN recognizes each region (Chinese model covers CJK + English)
 *   3. Low-confidence / Japanese regions → manga-ocr fallback
 *   4. Low-confidence Korean → dedicated Korean model fallback
 */

import * as ort from "onnxruntime-node";
import sharp from "sharp";
import path from "path";
import fs from "fs";

// ============================================================
// Types
// ============================================================

interface TextBox {
  box: [number, number, number, number]; // x, y, w, h in original image coords
  score: number;
}

interface OcrRegionResult {
  text: string;
  confidence: number;
  box: [number, number, number, number];
  lang: string;
}

// ============================================================
// Module-level state
// ============================================================

let detSession: ort.InferenceSession | null = null;

const recSessions = new Map<string, ort.InferenceSession>();
const recDicts = new Map<string, string[]>();

let mangaEncoderSession: ort.InferenceSession | null = null;
let mangaDecoderSession: ort.InferenceSession | null = null;
let mangaVocab: string[] = [];

// ============================================================
// Initialization
// ============================================================

export async function initOCR(modelsDir: string): Promise<void> {
  // Detection model
  const detPath = path.join(modelsDir, "paddleocr-det.onnx");
  if (fs.existsSync(detPath)) {
    detSession = await ort.InferenceSession.create(detPath);
    console.log("PaddleOCR detection model loaded");
  } else {
    console.warn("PaddleOCR detection model not found — OCR disabled");
    return;
  }

  // Recognition models (ch = CJK+English primary, ko = Korean fallback, en = English fallback)
  for (const lang of ["ch", "ko", "en"]) {
    const recPath = path.join(modelsDir, `paddleocr-rec-${lang}.onnx`);
    const dictPath = path.join(modelsDir, `paddleocr-dict-${lang}.txt`);
    if (fs.existsSync(recPath) && fs.existsSync(dictPath)) {
      recSessions.set(lang, await ort.InferenceSession.create(recPath));
      const lines = fs.readFileSync(dictPath, "utf-8").split("\n").filter((l) => l.length > 0);
      recDicts.set(lang, lines);
      console.log(`PaddleOCR rec model loaded: ${lang} (dict: ${lines.length} chars)`);
    }
  }

  // manga-ocr (Japanese fallback)
  const encPath = path.join(modelsDir, "manga-ocr-encoder.onnx");
  const decPath = path.join(modelsDir, "manga-ocr-decoder.onnx");
  const vocabPath = path.join(modelsDir, "manga-ocr-vocab.txt");
  if (fs.existsSync(encPath) && fs.existsSync(decPath) && fs.existsSync(vocabPath)) {
    mangaEncoderSession = await ort.InferenceSession.create(encPath);
    mangaDecoderSession = await ort.InferenceSession.create(decPath);
    mangaVocab = fs.readFileSync(vocabPath, "utf-8").split("\n");
    console.log(`manga-ocr loaded (vocab: ${mangaVocab.length} tokens)`);
  } else {
    console.warn("manga-ocr models not found — Japanese fallback disabled");
  }
}

// ============================================================
// PaddleOCR Text Detection (DBNet)
// ============================================================

const DET_LIMIT_SIDE = 960;
const DET_THRESH = 0.3;
const DET_BOX_THRESH = 0.6;
const DET_MIN_AREA = 9;
const DET_UNCLIP_RATIO = 1.5;

async function detectTextRegions(imageBuffer: Buffer): Promise<TextBox[]> {
  if (!detSession) return [];

  const meta = await sharp(imageBuffer).metadata();
  const origW = meta.width!;
  const origH = meta.height!;

  // Resize longest side to DET_LIMIT_SIDE, dims must be multiples of 32
  const ratio = Math.min(DET_LIMIT_SIDE / Math.max(origW, origH), 1.0);
  let newW = Math.max(32, Math.ceil((origW * ratio) / 32) * 32);
  let newH = Math.max(32, Math.ceil((origH * ratio) / 32) * 32);

  const scaleX = origW / newW;
  const scaleY = origH / newH;

  const { data } = await sharp(imageBuffer)
    .resize(newW, newH, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  // ImageNet normalization (RGB)
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const pixels = newH * newW;
  const float32 = new Float32Array(3 * pixels);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < pixels; i++) {
      float32[c * pixels + i] = (data[i * 3 + c] / 255.0 - mean[c]) / std[c];
    }
  }

  const tensor = new ort.Tensor("float32", float32, [1, 3, newH, newW]);
  const results = await detSession.run({ [detSession.inputNames[0]]: tensor });
  const output = results[detSession.outputNames[0]];
  const scoreMap = output.data as Float32Array;

  let scoreMin = Infinity, scoreMax = -Infinity, scoreSum = 0;
  for (let i = 0; i < scoreMap.length; i++) {
    if (scoreMap[i] < scoreMin) scoreMin = scoreMap[i];
    if (scoreMap[i] > scoreMax) scoreMax = scoreMap[i];
    scoreSum += scoreMap[i];
  }
  console.log(`[OCR-det] output shape=${JSON.stringify(output.dims)} scoreRange=[${scoreMin.toFixed(4)}, ${scoreMax.toFixed(4)}] mean=${(scoreSum / scoreMap.length).toFixed(4)}`);

  // Threshold → binary
  const binary = new Uint8Array(pixels);
  for (let i = 0; i < pixels; i++) {
    binary[i] = scoreMap[i] >= DET_THRESH ? 1 : 0;
  }

  // Connected components (DFS with stack for O(1) per op)
  const labels = new Int32Array(pixels);
  let numLabels = 0;
  const stack: number[] = [];

  for (let i = 0; i < pixels; i++) {
    if (binary[i] === 1 && labels[i] === 0) {
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
  }

  let binaryCount = 0;
  for (let i = 0; i < pixels; i++) if (binary[i]) binaryCount++;
  console.log(`[OCR-det] binary pixels=${binaryCount}/${pixels} (${(binaryCount/pixels*100).toFixed(2)}%), components=${numLabels}`);

  // Extract bounding boxes per component
  const comps = new Array<{ minX: number; minY: number; maxX: number; maxY: number; area: number; scoreSum: number }>(numLabels);
  for (let i = 0; i < numLabels; i++) {
    comps[i] = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity, area: 0, scoreSum: 0 };
  }
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const idx = y * newW + x;
      const label = labels[idx];
      if (label > 0) {
        const c = comps[label - 1];
        if (x < c.minX) c.minX = x;
        if (y < c.minY) c.minY = y;
        if (x > c.maxX) c.maxX = x;
        if (y > c.maxY) c.maxY = y;
        c.area++;
        c.scoreSum += scoreMap[idx];
      }
    }
  }

  let tooSmall = 0, lowScore = 0;
  const boxes: TextBox[] = [];
  for (const c of comps) {
    if (c.area < DET_MIN_AREA) { tooSmall++; continue; }
    const avgScore = c.scoreSum / c.area;
    if (avgScore < DET_BOX_THRESH) { lowScore++; continue; }

    // Expand box
    const bw = c.maxX - c.minX + 1;
    const bh = c.maxY - c.minY + 1;
    const ex = (bw * (DET_UNCLIP_RATIO - 1)) / 2;
    const ey = (bh * (DET_UNCLIP_RATIO - 1)) / 2;

    const x = Math.max(0, Math.round((c.minX - ex) * scaleX));
    const y = Math.max(0, Math.round((c.minY - ey) * scaleY));
    const w = Math.min(origW - x, Math.round((bw + ex * 2) * scaleX));
    const h = Math.min(origH - y, Math.round((bh + ey * 2) * scaleY));

    if (w > 0 && h > 0) {
      boxes.push({ box: [x, y, w, h], score: avgScore });
    }
  }

  console.log(`[OCR-det] filtered: tooSmall=${tooSmall}, lowScore=${lowScore}, passed=${boxes.length}`);
  return boxes;
}

// ============================================================
// PaddleOCR Text Recognition (CRNN + CTC)
// ============================================================

const REC_HEIGHT = 48;
const REC_MAX_WIDTH = 320;

async function recognizeRegion(
  imageBuffer: Buffer,
  box: [number, number, number, number],
  lang: string,
): Promise<{ text: string; confidence: number }> {
  const session = recSessions.get(lang);
  const dict = recDicts.get(lang);
  if (!session || !dict) return { text: "", confidence: 0 };

  // Crop
  const meta = await sharp(imageBuffer).metadata();
  const imgW = meta.width!;
  const imgH = meta.height!;
  const left = Math.max(0, Math.min(box[0], imgW - 1));
  const top = Math.max(0, Math.min(box[1], imgH - 1));
  const width = Math.max(1, Math.min(box[2], imgW - left));
  const height = Math.max(1, Math.min(box[3], imgH - top));

  const cropped = await sharp(imageBuffer).extract({ left, top, width, height }).toBuffer();

  // Resize to fixed height, proportional width
  const targetW = Math.max(1, Math.min(Math.round(REC_HEIGHT * (width / height)), REC_MAX_WIDTH));

  const { data } = await sharp(cropped)
    .resize(targetW, REC_HEIGHT, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  // Normalize to [-1, 1]
  const pixels = REC_HEIGHT * targetW;
  const float32 = new Float32Array(3 * pixels);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < pixels; i++) {
      float32[c * pixels + i] = (data[i * 3 + c] / 255.0 - 0.5) / 0.5;
    }
  }

  const tensor = new ort.Tensor("float32", float32, [1, 3, REC_HEIGHT, targetW]);
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const output = results[session.outputNames[0]];

  return ctcDecode(output.data as Float32Array, output.dims as number[], dict);
}

function ctcDecode(
  logits: Float32Array,
  dims: number[],
  dict: string[],
): { text: string; confidence: number } {
  // Output shape: [1, T, C] or [T, 1, C]
  let T: number, C: number;
  if (dims.length === 3 && dims[0] === 1) {
    T = dims[1]; C = dims[2];
  } else if (dims.length === 3) {
    T = dims[0]; C = dims[2];
  } else {
    return { text: "", confidence: 0 };
  }

  // PaddleOCR: class 0 = blank, class i (i>0) = dict[i-1]
  const BLANK = 0;
  let text = "";
  let confSum = 0;
  let confCount = 0;
  let prevIdx = -1;

  for (let t = 0; t < T; t++) {
    const base = t * C;
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let c = 0; c < C; c++) {
      if (logits[base + c] > maxVal) {
        maxVal = logits[base + c];
        maxIdx = c;
      }
    }

    // Softmax probability of max class (for confidence)
    let expSum = 0;
    for (let c = 0; c < C; c++) expSum += Math.exp(logits[base + c] - maxVal);
    const prob = 1.0 / expSum;

    if (maxIdx !== BLANK && maxIdx !== prevIdx) {
      const charIdx = maxIdx - 1;
      if (charIdx >= 0 && charIdx < dict.length) {
        text += dict[charIdx];
      }
      confSum += prob;
      confCount++;
    }
    prevIdx = maxIdx;
  }

  return { text: text.trim(), confidence: confCount > 0 ? confSum / confCount : 0 };
}

// ============================================================
// manga-ocr (Japanese fallback)
// ============================================================

const MANGA_MAX_LEN = 300;
const MANGA_START = 2; // [CLS]
const MANGA_END = 3; // [SEP]
const MANGA_PAD = 0;
const MANGA_MASK = 4;

async function mangaOcrRecognize(regionBuffer: Buffer): Promise<string> {
  if (!mangaEncoderSession || !mangaDecoderSession || mangaVocab.length === 0) return "";

  // Preprocess: 224x224, normalize to [-1, 1]
  const size = 224;
  const { data } = await sharp(regionBuffer)
    .resize(size, size, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32 = new Float32Array(3 * size * size);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < size * size; i++) {
      float32[c * size * size + i] = (data[i * 3 + c] / 255.0 - 0.5) / 0.5;
    }
  }

  // Encode
  const pixelValues = new ort.Tensor("float32", float32, [1, 3, size, size]);
  const encResult = await mangaEncoderSession.run({
    [mangaEncoderSession.inputNames[0]]: pixelValues,
  });
  const encoderHidden = encResult[mangaEncoderSession.outputNames[0]];

  // Autoregressive decode (greedy)
  const tokenIds: number[] = [MANGA_START];

  for (let step = 0; step < MANGA_MAX_LEN; step++) {
    const inputIds = new ort.Tensor(
      "int64",
      BigInt64Array.from(tokenIds.map(BigInt)),
      [1, tokenIds.length],
    );

    // Build decoder feed — use session.inputNames to be safe
    const decoderFeed: Record<string, ort.Tensor> = {};
    for (const name of mangaDecoderSession.inputNames) {
      if (name === "input_ids") decoderFeed[name] = inputIds;
      else if (name.includes("encoder_hidden") || name.includes("encoder_output")) decoderFeed[name] = encoderHidden;
      else if (name.includes("attention_mask") && name.includes("encoder")) {
        // All-ones mask matching encoder sequence length
        const seqLen = encoderHidden.dims[1] as number;
        decoderFeed[name] = new ort.Tensor("int64", BigInt64Array.from(Array(seqLen).fill(1n)), [1, seqLen]);
      }
    }

    const decResult = await mangaDecoderSession.run(decoderFeed);
    const logits = decResult[mangaDecoderSession.outputNames[0]];
    const logitsData = logits.data as Float32Array;
    const vocabSize = logits.dims[2] as number;
    const seqLen = logits.dims[1] as number;

    // Argmax on last position
    const offset = (seqLen - 1) * vocabSize;
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let i = 0; i < vocabSize; i++) {
      if (logitsData[offset + i] > maxVal) {
        maxVal = logitsData[offset + i];
        maxIdx = i;
      }
    }

    if (maxIdx === MANGA_END) break;
    tokenIds.push(maxIdx);
  }

  // Decode: skip special tokens, join, remove wordpiece "##" prefix
  return tokenIds
    .slice(1)
    .filter((id) => id !== MANGA_START && id !== MANGA_END && id !== MANGA_PAD && id !== MANGA_MASK)
    .map((id) => (id < mangaVocab.length ? mangaVocab[id] : ""))
    .join("")
    .replace(/##/g, "");
}

// ============================================================
// Language Detection & Fallback
// ============================================================

function getLanguageRatio(text: string): { jp: number; kr: number; en: number; cn: number } {
  const jp = (text.match(/[ぁ-んァ-ヶ]/g) || []).length;
  const kr = (text.match(/[가-힣]/g) || []).length;
  const en = (text.match(/[a-zA-Z]/g) || []).length;
  const cn = (text.match(/[\u4e00-\u9fff]/g) || []).length; // CJK ideographs (shared JP/CN)
  const total = text.replace(/\s/g, "").length || 1;
  return { jp: jp / total, kr: kr / total, en: en / total, cn: cn / total };
}

function detectLang(text: string): string {
  const r = getLanguageRatio(text);
  if (r.jp > 0.3 || (r.cn > 0.2 && r.jp > 0)) return "jp";
  if (r.kr > 0.3) return "kr";
  if (r.cn > 0.3) return "cn";
  return "en";
}

function shouldFallback(text: string, confidence: number): boolean {
  if (!text || text.trim().length === 0) return true;
  if (confidence < 0.8) return true;
  if (text.length <= 2 && confidence < 0.9) return true;

  const r = getLanguageRatio(text);
  if (r.jp > 0 && r.jp < 0.5 && r.cn === 0) return true;

  // Mostly special characters / noise
  const clean = text.replace(/[\p{L}\p{N}\s]/gu, "");
  if (clean.length > text.length * 0.5) return true;

  return false;
}

// ============================================================
// Main OCR Pipeline
// ============================================================

export async function performOCR(
  imageBuffer: Buffer,
  searchStrings: string[],
): Promise<{ found: string[]; fullText: string }> {
  if (!detSession || recSessions.size === 0) {
    console.log(`[OCR] SKIP — detSession=${!!detSession}, recSessions=${recSessions.size}`);
    return { found: [], fullText: "" };
  }

  // 1. Detect text regions
  const textBoxes = await detectTextRegions(imageBuffer);
  if (textBoxes.length === 0) {
    console.log("[OCR] detected 0 text regions");
    return { found: [], fullText: "" };
  }
  console.log(`[OCR] detected ${textBoxes.length} text regions`);

  // 2. Primary recognition with Chinese model (covers CJK + English)
  const primary: OcrRegionResult[] = [];
  for (const tb of textBoxes) {
    const res = await recognizeRegion(imageBuffer, tb.box, "ch");
    primary.push({
      text: res.text,
      confidence: res.confidence,
      box: tb.box,
      lang: detectLang(res.text),
    });
  }

  // 3. Fallback where needed
  const finalTexts: string[] = [];
  for (const pr of primary) {
    if (!shouldFallback(pr.text, pr.confidence)) {
      finalTexts.push(pr.text);
      console.log(`[OCR] region "${pr.text}" conf=${pr.confidence.toFixed(3)} lang=${pr.lang} → OK`);
      continue;
    }

    console.log(`[OCR] region "${pr.text}" conf=${pr.confidence.toFixed(3)} lang=${pr.lang} → FALLBACK`);
    let fallbackText: string | null = null;

    if (pr.lang === "jp" || pr.lang === "cn") {
      // manga-ocr for Japanese / CJK
      if (mangaEncoderSession) {
        const cropped = await safeCrop(imageBuffer, pr.box);
        const result = await mangaOcrRecognize(cropped);
        if (result.length > 0) {
          fallbackText = result;
          console.log(`[OCR]   manga-ocr → "${result}"`);
        }
      }
    } else if (pr.lang === "kr" && recSessions.has("ko")) {
      const kr = await recognizeRegion(imageBuffer, pr.box, "ko");
      if (kr.text && kr.confidence > pr.confidence) {
        fallbackText = kr.text;
        console.log(`[OCR]   korean-rec → "${kr.text}" conf=${kr.confidence.toFixed(3)}`);
      }
    } else if (pr.lang === "en" && recSessions.has("en")) {
      const en = await recognizeRegion(imageBuffer, pr.box, "en");
      if (en.text && en.confidence > pr.confidence) {
        fallbackText = en.text;
        console.log(`[OCR]   english-rec → "${en.text}" conf=${en.confidence.toFixed(3)}`);
      }
    }

    finalTexts.push(fallbackText || pr.text);
  }

  // 4. Combine
  const fullText = finalTexts.join("\n");

  // 5. Match search strings
  const found = searchStrings.filter((s) => {
    if (s.includes("\t")) {
      return s.split("\t").every((part) => fullText.includes(part));
    }
    return fullText.includes(s);
  });

  return { found, fullText };
}

// ============================================================
// Helpers
// ============================================================

async function safeCrop(imageBuffer: Buffer, box: [number, number, number, number]): Promise<Buffer> {
  const meta = await sharp(imageBuffer).metadata();
  const imgW = meta.width!;
  const imgH = meta.height!;
  const left = Math.max(0, Math.min(box[0], imgW - 1));
  const top = Math.max(0, Math.min(box[1], imgH - 1));
  const width = Math.max(1, Math.min(box[2], imgW - left));
  const height = Math.max(1, Math.min(box[3], imgH - top));
  return sharp(imageBuffer).extract({ left, top, width, height }).toBuffer();
}
