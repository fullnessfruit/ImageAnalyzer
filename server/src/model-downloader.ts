import fs from "fs";
import path from "path";
import https from "https";
import http from "http";

interface ModelInfo {
  url: string;
  filename: string;
}

const MODELS: Record<string, ModelInfo> = {
  clip: {
    url: "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/main/model.onnx",
    filename: "clip-vit-base-patch32.onnx",
  },
  arcface: {
    url: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    filename: "arcface-w600k-r50.onnx",
  },
  facedet: {
    url: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
    filename: "face-detection.onnx",
  },
  "yolo-person": {
    url: "https://huggingface.co/s1777/yolo-v8n-onnx/resolve/main/yolov8n.onnx",
    filename: "yolo-person.onnx",
  },
  // --- PaddleOCR ---
  "paddleocr-det": {
    url: "https://huggingface.co/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_det.onnx",
    filename: "paddleocr-det.onnx",
  },
  "paddleocr-rec-ch": {
    url: "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/rec.onnx",
    filename: "paddleocr-rec-ch.onnx",
  },
  "paddleocr-dict-ch": {
    url: "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/dict.txt",
    filename: "paddleocr-dict-ch.txt",
  },
  "paddleocr-rec-ko": {
    url: "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/korean/rec.onnx",
    filename: "paddleocr-rec-ko.onnx",
  },
  "paddleocr-dict-ko": {
    url: "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/korean/dict.txt",
    filename: "paddleocr-dict-ko.txt",
  },
  "paddleocr-rec-en": {
    url: "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/rec.onnx",
    filename: "paddleocr-rec-en.onnx",
  },
  "paddleocr-dict-en": {
    url: "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/dict.txt",
    filename: "paddleocr-dict-en.txt",
  },
  // --- manga-ocr (Japanese fallback) ---
  "manga-ocr-encoder": {
    url: "https://huggingface.co/l0wgear/manga-ocr-2025-onnx/resolve/main/encoder_model.onnx",
    filename: "manga-ocr-encoder.onnx",
  },
  "manga-ocr-decoder": {
    url: "https://huggingface.co/l0wgear/manga-ocr-2025-onnx/resolve/main/decoder_model.onnx",
    filename: "manga-ocr-decoder.onnx",
  },
  "manga-ocr-vocab": {
    url: "https://huggingface.co/l0wgear/manga-ocr-2025-onnx/resolve/main/vocab.txt",
    filename: "manga-ocr-vocab.txt",
  },
};

function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const request = (url.startsWith("https") ? https : http).get(url, (response) => {
      if (response.statusCode && response.statusCode >= 301 && response.statusCode <= 308 && response.statusCode !== 304) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          file.close();
          fs.unlinkSync(dest);
          downloadFile(redirectUrl, dest).then(resolve).catch(reject);
          return;
        }
      }
      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(dest);
        reject(new Error(`Download failed with status ${response.statusCode} for ${url}`));
        return;
      }
      const totalBytes = parseInt(response.headers["content-length"] || "0", 10);
      let downloadedBytes = 0;
      response.on("data", (chunk: Buffer) => {
        downloadedBytes += chunk.length;
        if (totalBytes > 0) {
          const pct = ((downloadedBytes / totalBytes) * 100).toFixed(1);
          process.stdout.write(`\rDownloading ${path.basename(dest)}: ${pct}%`);
        }
      });
      response.pipe(file);
      file.on("finish", () => {
        file.close();
        console.log(`\nDownloaded ${path.basename(dest)}`);
        resolve();
      });
    });
    request.on("error", (err) => {
      file.close();
      if (fs.existsSync(dest)) fs.unlinkSync(dest);
      reject(err);
    });
  });
}

export async function ensureModelsDownloaded(modelsDir: string): Promise<void> {
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
  }

  for (const [name, info] of Object.entries(MODELS)) {
    const dest = path.join(modelsDir, info.filename);
    if (fs.existsSync(dest)) {
      console.log(`Model ${info.filename} already exists, skipping.`);
      continue;
    }
    console.log(`Downloading ${name} model from ${info.url}...`);
    try {
      await downloadFile(info.url, dest);
    } catch (err) {
      console.error(`Failed to download ${name} model: ${err}`);
      console.error(`Please manually download from ${info.url} and place at ${dest}`);
    }
  }
}

export function getModelPath(modelsDir: string, modelName: keyof typeof MODELS): string {
  return path.join(modelsDir, MODELS[modelName].filename);
}
