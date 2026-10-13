/** 분석 요청과 결과를 나중에 같은 입력으로 검사할 수 있게 로컬 파일에 남긴다. */

import fs from "fs";
import path from "path";
import { PROJECT_ROOT } from "./config";

export type AnalysisLogEntry = Record<string, unknown> & {
  event: "image_request" | "image_result" | "job_result" | "job_delivery";
  mode: "worker" | "http" | "cli";
};

// AnnouncementAggregator의 다운로드 로그처럼 파일명과 행 시작에 ISO UTC 시각을 쓴다.
// PID까지 포함해 같은 초에 기동한 여러 워커가 한 파일에 쓰지 않도록 한다.
const STARTED_AT_UTC = `${new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5)}Z`;
const LOG_FILENAME = `analysis-log_${STARTED_AT_UTC}_${process.pid}.log`;

/** 한 이벤트를 한 줄에 쓰며, 원본 TSV의 줄바꿈은 JSON 문자열 안에서 이스케이프한다. */
export function appendAnalysisLog(entry: AnalysisLogEntry, projectRoot: string = PROJECT_ROOT): void {
  const timestamp = new Date().toISOString();
  const directory = path.join(projectRoot, "log");
  fs.mkdirSync(directory, { recursive: true });
  fs.appendFileSync(path.join(directory, LOG_FILENAME), `[${timestamp}] ${JSON.stringify({ processId: process.pid, ...entry })}\n`, "utf8");
}
