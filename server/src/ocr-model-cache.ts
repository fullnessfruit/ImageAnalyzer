/** Prepare a warning-free copy of the exported v5 OCR model without changing its inference graph. */
import crypto from "crypto";
import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-node";

const SESSION_OPTIONS: ort.InferenceSession.SessionOptions = { intraOpNumThreads: 4 };

function prunedModelPath(sourcePath: string): string {
  const runtimeVersion = require("onnxruntime-node/package.json").version as string;
  const digest = crypto.createHash("sha256")
    .update(fs.readFileSync(sourcePath))
    .update(`pruned-v1:${runtimeVersion}:${process.platform}:${process.arch}`)
    .digest("hex").slice(0, 16);
  return path.join(path.dirname(sourcePath), `ocr-rec-v5-pruned-${digest}.onnx`);
}

async function preparePrunedModel(sourcePath: string): Promise<string> {
  const prunedPath = prunedModelPath(sourcePath);
  if (fs.existsSync(prunedPath)) return prunedPath;

  // With graph rewrites disabled, exporting saves ONNX Runtime's graph cleanup only.
  // The temporary session hides its known unused-initializer messages; the real session
  // loads the cleaned copy with normal warning reporting.
  const tempPath = `${prunedPath}.${process.pid}.${crypto.randomUUID()}.tmp.onnx`;
  try {
    const preparation = await ort.InferenceSession.create(sourcePath, {
      ...SESSION_OPTIONS,
      graphOptimizationLevel: "disabled",
      optimizedModelFilePath: tempPath,
      logSeverityLevel: 3,
    });
    await preparation.release();
    if (!fs.existsSync(tempPath)) throw new Error("ONNX Runtime did not write the pruned model");
    try {
      fs.renameSync(tempPath, prunedPath);
    } catch (error) {
      // Another worker may have published the same source/runtime copy first.
      if (!fs.existsSync(prunedPath)) throw error;
    }
    return prunedPath;
  } finally {
    fs.rmSync(tempPath, { force: true });
  }
}

export async function loadV5Session(sourcePath: string): Promise<ort.InferenceSession> {
  try {
    return await ort.InferenceSession.create(await preparePrunedModel(sourcePath), SESSION_OPTIONS);
  } catch (error: any) {
    // Keep OCR available if the cache cannot be written or loaded.
    console.warn(`OCR v5 pruned model unavailable - source: ${sourcePath}, error: ${error.message}`);
    return ort.InferenceSession.create(sourcePath, { ...SESSION_OPTIONS, logSeverityLevel: 3 });
  }
}
