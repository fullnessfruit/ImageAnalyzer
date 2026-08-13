/**
 * config.json 로더.
 *
 * 서버, CLI, 워커가 같은 파일을 같은 규칙으로 읽어야 하므로 한 곳에 둔다. 세 곳에
 * 복사해 두면 기본값 병합 규칙이 조용히 갈라진다.
 *
 * 파일이 없거나 깨져도 기본값으로 동작한다. 설정을 못 읽었다고 분석을 멈출 이유가 없다.
 * 매 호출마다 다시 읽으므로 재시작 없이 반영된다.
 */

import path from "path";
import fs from "fs";
import { Config, DEFAULT_CONFIG } from "./analyze";

/** 리포 루트. `server/src`에서 두 단계 위. */
export const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

export function loadConfig(projectRoot: string = PROJECT_ROOT): Config {
  const configPath = path.join(projectRoot, "config.json");
  try {
    const raw = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    return {
      similarityThreshold: { ...DEFAULT_CONFIG.similarityThreshold, ...(raw.similarityThreshold ?? {}) },
      margin: { ...DEFAULT_CONFIG.margin, ...(raw.margin ?? {}) },
      ocr: { ...DEFAULT_CONFIG.ocr, ...(raw.ocr ?? {}) },
      wdTagger: { ...DEFAULT_CONFIG.wdTagger, ...(raw.wdTagger ?? {}) },
      characterAliases: { ...DEFAULT_CONFIG.characterAliases, ...(raw.characterAliases ?? {}) },
      candidates: { ...DEFAULT_CONFIG.candidates, ...(raw.candidates ?? {}) },
    };
  } catch (e: any) {
    console.warn(`Config load failed - path: ${configPath}, error: ${e.message} (기본값 사용)`);
    return DEFAULT_CONFIG;
  }
}
