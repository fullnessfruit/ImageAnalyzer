/**
 * ONNX 모델 자동 다운로드.
 *
 * HuggingFace는 /resolve/main/ 요청에 대해 307(상대경로 Location) 또는 302(절대 CDN URL)를
 * 반환한다. 상대경로를 그대로 재요청하면 "Invalid URL"로 실패하므로 반드시 기준 URL에
 * 상대 해석해야 한다. 또한 응답이 200으로 확정되기 전에는 파일 핸들을 열지 않는다 -
 * Windows에서 열린 핸들에 unlink를 시도하면 EPERM이 난다.
 */

import fs from "fs";
import path from "path";
import https from "https";
import http from "http";

interface ModelInfo {
  url: string;
  filename: string;
  /** 없어도 서버가 기동하는 모델은 optional. 해당 기능만 비활성화된다. */
  optional?: boolean;
}

const HF = "https://huggingface.co";

export const MODELS = {
  // --- OCR: PaddleOCR ---
  // chinese 사전은 한자 15565 + 히라가나 84 + 가타카나 89 + 라틴 52 + 숫자를 포함해
  // 일본어/중국어/영어를 모두 커버한다. korean 사전은 한글 11172 전용(한자·가나 없음).
  // 따라서 검색어의 문자 종류로 rec 모델을 고른다. 영어 전용 모델은 chinese와 중복이라 쓰지 않는다.
  "ocr-det": {
    url: `${HF}/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_det.onnx`,
    filename: "ocr-det.onnx",
  },
  "ocr-rec-ch": {
    url: `${HF}/monkt/paddleocr-onnx/resolve/main/languages/chinese/rec.onnx`,
    filename: "ocr-rec-ch.onnx",
  },
  "ocr-dict-ch": {
    url: `${HF}/monkt/paddleocr-onnx/resolve/main/languages/chinese/dict.txt`,
    filename: "ocr-dict-ch.txt",
  },
  "ocr-rec-ko": {
    url: `${HF}/monkt/paddleocr-onnx/resolve/main/languages/korean/rec.onnx`,
    filename: "ocr-rec-ko.onnx",
  },
  "ocr-dict-ko": {
    url: `${HF}/monkt/paddleocr-onnx/resolve/main/languages/korean/dict.txt`,
    filename: "ocr-dict-ko.txt",
  },

  // --- 실사 인물 ---
  // SCRFD는 5점 랜드마크를 함께 출력한다. ArcFace는 그 랜드마크로 정규 템플릿에
  // 워프된 얼굴로만 학습되었으므로 정합이 필수다.
  "face-det": {
    url: `${HF}/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx`,
    filename: "face-det.onnx",
  },
  arcface: {
    url: `${HF}/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx`,
    filename: "arcface-w600k-r50.onnx",
  },

  // --- 캐릭터 ---
  // 애니 얼굴/전신 검출기는 YOLOv8 export (imgsz 640, 단일 클래스).
  "anime-face-det": {
    url: `${HF}/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_s/model.onnx`,
    filename: "anime-face-det.onnx",
  },
  "anime-person-det": {
    url: `${HF}/deepghs/anime_person_detection/resolve/main/person_detect_v1.3_s/model.onnx`,
    filename: "anime-person-det.onnx",
  },
  // CCIP: 같은 캐릭터를 서로 다른 작가·화풍에 걸쳐 positive로 학습한 대조 모델.
  // 화풍이 바뀌어도 신원이 유지되므로 동인 일러스트 대응의 핵심이다.
  ccip: {
    url: `${HF}/deepghs/ccip_onnx/resolve/main/ccip-caformer-24-randaug-pruned/model_feat.onnx`,
    filename: "ccip-feat.onnx",
  },
  // WD-Tagger: 캐릭터 이름과 의류 태그를 동시에 출력. 캐릭터는 등록 0장으로 식별되고,
  // 의류 태그는 화풍·착용자에 불변인 의상 표현이 된다.
  "wd-tagger": {
    url: `${HF}/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx`,
    filename: "wd-tagger.onnx",
  },
  "wd-tags": {
    url: `${HF}/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv`,
    filename: "wd-tags.csv",
  },
} satisfies Record<string, ModelInfo>;

export type ModelName = keyof typeof MODELS;

const MAX_REDIRECTS = 10;

function fetchToFile(url: string, dest: string, redirectsLeft: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https:") ? https : http;
    const request = client.get(url, { headers: { "User-Agent": "ImageAnalyzer/1.0" } }, (response) => {
      const status = response.statusCode ?? 0;

      if (status >= 300 && status < 400 && status !== 304) {
        const location = response.headers.location;
        response.resume(); // 소켓 재사용을 위해 본문 폐기
        if (!location) {
          reject(new Error(`HTTP ${status} without Location header: ${url}`));
          return;
        }
        if (redirectsLeft <= 0) {
          reject(new Error(`Too many redirects: ${url}`));
          return;
        }
        // HF는 307에 상대경로를 준다. 기준 URL에 상대 해석해야 한다.
        const next = new URL(location, url).toString();
        fetchToFile(next, dest, redirectsLeft - 1).then(resolve).catch(reject);
        return;
      }

      if (status !== 200) {
        response.resume();
        reject(new Error(`HTTP ${status}: ${url}`));
        return;
      }

      // 200 확정 후에만 핸들을 연다.
      const tmp = `${dest}.part`;
      const file = fs.createWriteStream(tmp);
      const totalBytes = parseInt(response.headers["content-length"] || "0", 10);
      let downloaded = 0;
      let lastPct = -1;

      response.on("data", (chunk: Buffer) => {
        downloaded += chunk.length;
        if (totalBytes > 0) {
          const pct = Math.floor((downloaded / totalBytes) * 100);
          if (pct !== lastPct) {
            lastPct = pct;
            process.stdout.write(`\r  ${path.basename(dest)}: ${pct}%`);
          }
        }
      });

      response.on("error", (err) => {
        file.destroy();
        fs.rmSync(tmp, { force: true });
        reject(err);
      });

      response.pipe(file);

      file.on("error", (err) => {
        file.destroy();
        fs.rmSync(tmp, { force: true });
        reject(err);
      });

      // rename은 close 완료 후에만 안전하다.
      file.on("close", () => {
        if (totalBytes > 0 && downloaded !== totalBytes) {
          fs.rmSync(tmp, { force: true });
          reject(new Error(`Incomplete download (${downloaded}/${totalBytes}): ${url}`));
          return;
        }
        fs.renameSync(tmp, dest);
        process.stdout.write(`\r  ${path.basename(dest)}: ${(downloaded / 1048576).toFixed(1)} MB\n`);
        resolve();
      });
    });

    request.on("error", reject);
    request.setTimeout(120000, () => {
      request.destroy(new Error(`Timeout: ${url}`));
    });
  });
}

export async function ensureModelsDownloaded(modelsDir: string): Promise<void> {
  fs.mkdirSync(modelsDir, { recursive: true });

  const failed: string[] = [];
  for (const [name, info] of Object.entries(MODELS) as [string, ModelInfo][]) {
    const dest = path.join(modelsDir, info.filename);
    if (fs.existsSync(dest) && fs.statSync(dest).size > 0) continue;

    console.log(`Downloading ${name}...`);
    try {
      await fetchToFile(info.url, dest, MAX_REDIRECTS);
    } catch (err: any) {
      failed.push(name);
      console.error(`Download failed - model: ${name}, url: ${info.url}, error: ${err.message}`);
    }
  }

  if (failed.length > 0) {
    console.error(`Model download incomplete - failed: ${failed.join(", ")} (해당 기능이 비활성화된다)`);
  } else {
    console.log(`All models present - count: ${Object.keys(MODELS).length}, dir: ${modelsDir}`);
  }
}

export function getModelPath(modelsDir: string, name: ModelName): string {
  return path.join(modelsDir, MODELS[name].filename);
}
