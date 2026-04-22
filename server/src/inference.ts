import * as ort from "onnxruntime-node";
import sharp from "sharp";
import { getModelPath } from "./model-downloader";

let clipSession: ort.InferenceSession | null = null;
let arcfaceSession: ort.InferenceSession | null = null;
let faceDetSession: ort.InferenceSession | null = null;
let yoloSession: ort.InferenceSession | null = null;
let yoloPersonSession: ort.InferenceSession | null = null;

export async function loadModels(modelsDir: string): Promise<void> {
  const clipPath = getModelPath(modelsDir, "clip");
  const arcfacePath = getModelPath(modelsDir, "arcface");
  const faceDetPath = getModelPath(modelsDir, "facedet");

  try {
    clipSession = await ort.InferenceSession.create(clipPath);
    console.log("CLIP model loaded");
  } catch (e) {
    console.error(`Failed to load CLIP model: ${e}`);
  }

  try {
    arcfaceSession = await ort.InferenceSession.create(arcfacePath);
    console.log("ArcFace model loaded");
  } catch (e) {
    console.error(`Failed to load ArcFace model: ${e}`);
  }

  try {
    faceDetSession = await ort.InferenceSession.create(faceDetPath);
    console.log("Face detection model loaded");
  } catch (e) {
    console.error(`Failed to load face detection model: ${e}`);
  }

  const yoloPath = require("path").join(modelsDir, "yolo-characters.onnx");
  if (require("fs").existsSync(yoloPath)) {
    try {
      yoloSession = await ort.InferenceSession.create(yoloPath);
      console.log("YOLO model loaded");
    } catch (e) {
      console.error(`Failed to load YOLO model: ${e}`);
    }
  } else {
    console.warn("YOLO model not found at " + yoloPath + " — character detection disabled");
  }

  const yoloPersonPath = getModelPath(modelsDir, "yolo-person");
  if (require("fs").existsSync(yoloPersonPath)) {
    try {
      yoloPersonSession = await ort.InferenceSession.create(yoloPersonPath);
      console.log("YOLO person model loaded");
    } catch (e) {
      console.error(`Failed to load YOLO person model: ${e}`);
    }
  } else {
    console.warn("YOLO person model not found at " + yoloPersonPath + " — person detection disabled");
  }
}

// --- Image preprocessing ---

async function preprocessForCLIP(imageBuffer: Buffer): Promise<ort.Tensor> {
  const size = 224;
  const { data, info } = await sharp(imageBuffer)
    .resize(size, size, { fit: "cover" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32 = new Float32Array(3 * size * size);
  const mean = [0.48145466, 0.4578275, 0.40821073];
  const std = [0.26862954, 0.26130258, 0.27577711];

  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < size * size; i++) {
      float32[c * size * size + i] = (data[i * 3 + c] / 255.0 - mean[c]) / std[c];
    }
  }
  return new ort.Tensor("float32", float32, [1, 3, size, size]);
}

async function preprocessForArcFace(imageBuffer: Buffer): Promise<ort.Tensor> {
  const size = 112;
  const { data } = await sharp(imageBuffer)
    .resize(size, size, { fit: "cover" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32 = new Float32Array(3 * size * size);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < size * size; i++) {
      float32[c * size * size + i] = (data[i * 3 + c] - 127.5) / 127.5;
    }
  }
  return new ort.Tensor("float32", float32, [1, 3, size, size]);
}

async function preprocessForYOLO(imageBuffer: Buffer, targetSize: number = 640): Promise<{ tensor: ort.Tensor; scale: { sx: number; sy: number }; origSize: { w: number; h: number } }> {
  const meta = await sharp(imageBuffer).metadata();
  const origW = meta.width!;
  const origH = meta.height!;

  const { data } = await sharp(imageBuffer)
    .resize(targetSize, targetSize, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32 = new Float32Array(3 * targetSize * targetSize);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < targetSize * targetSize; i++) {
      float32[c * targetSize * targetSize + i] = data[i * 3 + c] / 255.0;
    }
  }

  return {
    tensor: new ort.Tensor("float32", float32, [1, 3, targetSize, targetSize]),
    scale: { sx: origW / targetSize, sy: origH / targetSize },
    origSize: { w: origW, h: origH },
  };
}

async function preprocessForFaceDet(imageBuffer: Buffer, targetSize: number = 640): Promise<{ tensor: ort.Tensor; ratio: number; padX: number; padY: number }> {
  const meta = await sharp(imageBuffer).metadata();
  const origW = meta.width!;
  const origH = meta.height!;

  // Letterbox: resize preserving aspect ratio, pad remainder with mean color
  const ratio = Math.min(targetSize / origW, targetSize / origH);
  const newW = Math.round(origW * ratio);
  const newH = Math.round(origH * ratio);
  const padX = Math.round((targetSize - newW) / 2);
  const padY = Math.round((targetSize - newH) / 2);

  const { data } = await sharp(imageBuffer)
    .resize(newW, newH)
    .extend({
      top: padY,
      bottom: targetSize - newH - padY,
      left: padX,
      right: targetSize - newW - padX,
      background: { r: 127, g: 127, b: 127 },
    })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32 = new Float32Array(3 * targetSize * targetSize);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < targetSize * targetSize; i++) {
      float32[c * targetSize * targetSize + i] = (data[i * 3 + c] - 127.5) / 128.0;
    }
  }

  return {
    tensor: new ort.Tensor("float32", float32, [1, 3, targetSize, targetSize]),
    ratio,
    padX,
    padY,
  };
}

// --- Inference functions ---

export async function extractCLIPEmbedding(imageBuffer: Buffer): Promise<Float32Array | null> {
  if (!clipSession) return null;
  const input = await preprocessForCLIP(imageBuffer);
  const inputName = clipSession.inputNames.find((n) => n.toLowerCase().includes("pixel") || n.toLowerCase().includes("image")) || clipSession.inputNames[0];
  const results = await clipSession.run({ [inputName]: input });
  const outputName = clipSession.outputNames.find((n) => n.toLowerCase().includes("image") || n.toLowerCase().includes("embed")) || clipSession.outputNames[0];
  const output = results[outputName];
  return new Float32Array(output.data as Float32Array);
}

export async function extractArcFaceEmbedding(imageBuffer: Buffer): Promise<Float32Array | null> {
  if (!arcfaceSession) return null;
  const input = await preprocessForArcFace(imageBuffer);
  const results = await arcfaceSession.run({ [arcfaceSession.inputNames[0]]: input });
  const output = results[arcfaceSession.outputNames[0]];
  return new Float32Array(output.data as Float32Array);
}

export interface Detection {
  box: [number, number, number, number]; // x, y, w, h
  confidence: number;
  classId: number;
}

export async function detectWithYOLO(imageBuffer: Buffer, confThreshold: number = 0.25): Promise<Detection[]> {
  if (!yoloSession) return [];
  const { tensor, scale } = await preprocessForYOLO(imageBuffer);
  const results = await yoloSession.run({ [yoloSession.inputNames[0]]: tensor });
  const output = results[yoloSession.outputNames[0]];
  return parseYOLOOutput(output, scale, confThreshold);
}

function parseYOLOOutput(output: ort.Tensor, scale: { sx: number; sy: number }, confThreshold: number): Detection[] {
  const data = output.data as Float32Array;
  const [, numFields, numBoxes] = output.dims as number[];
  const detections: Detection[] = [];

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    let maxConf = 0;
    let maxClassId = 0;
    for (let c = 0; c < numFields - 4; c++) {
      const conf = data[(4 + c) * numBoxes + i];
      if (conf > maxConf) {
        maxConf = conf;
        maxClassId = c;
      }
    }

    if (maxConf >= confThreshold) {
      detections.push({
        box: [
          Math.round((cx - w / 2) * scale.sx),
          Math.round((cy - h / 2) * scale.sy),
          Math.round(w * scale.sx),
          Math.round(h * scale.sy),
        ],
        confidence: maxConf,
        classId: maxClassId,
      });
    }
  }

  return nms(detections, 0.45);
}

/** Detect persons using COCO-pretrained YOLO. Filters for class 0 (person). */
export async function detectPersons(imageBuffer: Buffer, confThreshold: number = 0.25): Promise<Detection[]> {
  if (!yoloPersonSession) return [];
  const { tensor, scale, origSize } = await preprocessForYOLO(imageBuffer);
  const results = await yoloPersonSession.run({ [yoloPersonSession.inputNames[0]]: tensor });
  const output = results[yoloPersonSession.outputNames[0]];

  // Some pretrained YOLO exports output normalized [0,1] coords instead of pixel coords.
  // Detect by checking bbox magnitude: pixel coords are in hundreds, normalized are < 2.
  const data = output.data as Float32Array;
  const [, , numBoxes] = output.dims as number[];
  let maxVal = 0;
  for (let i = 0; i < Math.min(numBoxes, 100); i++) {
    for (let j = 0; j < 4; j++) {
      const v = Math.abs(data[j * numBoxes + i]);
      if (v > maxVal) maxVal = v;
    }
  }
  // Normalized coords → scale directly to original size; pixel coords → use normal scale
  const effectiveScale = maxVal < 2
    ? { sx: origSize.w, sy: origSize.h }
    : scale;

  const all = parseYOLOOutput(output, effectiveScale, confThreshold);
  return all.filter((d) => d.classId === 0);
}

export async function detectFaces(imageBuffer: Buffer, confThreshold: number = 0.5): Promise<Detection[]> {
  if (!faceDetSession) return [];
  const targetSize = 640;
  const { tensor, ratio, padX, padY } = await preprocessForFaceDet(imageBuffer, targetSize);
  const results = await faceDetSession.run({ [faceDetSession.inputNames[0]]: tensor });

  // SCRFD (InsightFace det_10g) outputs 9 tensors across 3 FPN levels (stride 8, 16, 32):
  //   3x score [N, 1], 3x bbox [N, 4], 3x landmarks [N, 10]
  // Group by column count, sort by anchor count descending (stride 8 → 16 → 32)
  const outputs = faceDetSession.outputNames.map((n) => results[n]);
  const scoreOutputs = outputs.filter((o) => o.dims[1] === 1).sort((a, b) => (b.dims[0] as number) - (a.dims[0] as number));
  const bboxOutputs = outputs.filter((o) => o.dims[1] === 4).sort((a, b) => (b.dims[0] as number) - (a.dims[0] as number));

  if (scoreOutputs.length !== 3 || bboxOutputs.length !== 3) return [];

  const strides = [8, 16, 32];
  const detections: Detection[] = [];
  const invRatio = 1 / ratio;

  for (let si = 0; si < 3; si++) {
    const stride = strides[si];
    const scores = scoreOutputs[si].data as Float32Array;
    const bboxes = bboxOutputs[si].data as Float32Array;
    const fmSize = targetSize / stride;
    const numAnchorsPerCell = 2;

    let anchorIdx = 0;
    for (let row = 0; row < fmSize; row++) {
      for (let col = 0; col < fmSize; col++) {
        for (let a = 0; a < numAnchorsPerCell; a++) {
          const score = scores[anchorIdx];
          if (score >= confThreshold) {
            const anchorX = col * stride;
            const anchorY = row * stride;
            // Map from padded model coords back to original image coords
            const x1 = (anchorX - bboxes[anchorIdx * 4 + 0] * stride - padX) * invRatio;
            const y1 = (anchorY - bboxes[anchorIdx * 4 + 1] * stride - padY) * invRatio;
            const x2 = (anchorX + bboxes[anchorIdx * 4 + 2] * stride - padX) * invRatio;
            const y2 = (anchorY + bboxes[anchorIdx * 4 + 3] * stride - padY) * invRatio;

            detections.push({
              box: [Math.round(x1), Math.round(y1), Math.round(x2 - x1), Math.round(y2 - y1)],
              confidence: score,
              classId: -1,
            });
          }
          anchorIdx++;
        }
      }
    }
  }

  return nms(detections, 0.4);
}

function nms(detections: Detection[], iouThreshold: number): Detection[] {
  detections.sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];

  for (const det of detections) {
    let dominated = false;
    for (const k of kept) {
      if (iou(det.box, k.box) > iouThreshold) {
        dominated = true;
        break;
      }
    }
    if (!dominated) kept.push(det);
  }
  return kept;
}

function iou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const bx2 = b[0] + b[2], by2 = b[1] + b[3];
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function cropRegion(imageBuffer: Buffer, box: [number, number, number, number]): Promise<Buffer> {
  const meta = await sharp(imageBuffer).metadata();
  const imgW = meta.width!;
  const imgH = meta.height!;

  const left = Math.max(0, Math.min(box[0], imgW - 1));
  const top = Math.max(0, Math.min(box[1], imgH - 1));
  const width = Math.max(1, Math.min(box[2], imgW - left));
  const height = Math.max(1, Math.min(box[3], imgH - top));

  return sharp(imageBuffer)
    .extract({ left, top, width, height })
    .toBuffer();
}

/**
 * Extract costume region from a character detection by masking out the face area.
 * Finds the face whose center falls within the character box, then overlays
 * a gray rectangle over the face (+30% padding above for hair) so CLIP
 * embedding reflects only the costume.
 * Returns null if no face is found within the character box.
 */
export async function extractCostumeRegion(
  imageBuffer: Buffer,
  characterBox: [number, number, number, number],
  faceBoxes: Detection[]
): Promise<Buffer | null> {
  const [cx, cy, cw, ch] = characterBox;

  const face = faceBoxes.find((f) => {
    const fcx = f.box[0] + f.box[2] / 2;
    const fcy = f.box[1] + f.box[3] / 2;
    return fcx >= cx && fcx <= cx + cw && fcy >= cy && fcy <= cy + ch;
  });

  if (!face) return null;

  const charCrop = await cropRegion(imageBuffer, characterBox);

  const [fx, fy, fw, fh] = face.box;
  const padTop = Math.round(fh * 0.3);
  const maskLeft = Math.max(0, Math.round(fx - cx));
  const maskTop = Math.max(0, Math.round(fy - cy - padTop));
  const maskWidth = Math.max(1, Math.min(Math.round(fw), cw - maskLeft));
  const maskHeight = Math.max(1, Math.min(Math.round(fh + padTop), ch - maskTop));

  const mask = await sharp({
    create: {
      width: maskWidth,
      height: maskHeight,
      channels: 3,
      background: { r: 128, g: 128, b: 128 },
    },
  })
    .png()
    .toBuffer();

  return sharp(charCrop)
    .composite([{ input: mask, left: maskLeft, top: maskTop }])
    .toBuffer();
}
