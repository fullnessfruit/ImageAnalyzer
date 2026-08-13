/**
 * OCR 워커 - NAT 안쪽에서 브로커를 당겨 일하는 모드.
 *
 * 왜 이 모드가 있나
 * 크롬 확장은 공인 주소를 가진 **매우 사양이 낮은 머신**에서 돌고 이 프로그램은 집 노트북
 * (NAT 안)에서 돈다. 확장이 노트북을 호출할 수 없으므로 방향을 뒤집는다. 노트북이 브로커
 * (AnnouncementAggregator/ocr-broker)를 주기적으로 당겨 작업을 가져가고 결과를 올린다.
 * HTTP 서버 모드(`npm run server`)는 그대로 남아 있고, 같은 기계에서 부를 수 있을 때 쓴다.
 *
 * OCR만 쓴다
 * 확장이 묻는 것은 "이 이미지에 이 문자열들이 있는가" 하나뿐이라 `performOCR`을 직접 부른다.
 * 비전 모델 6개(약 772MB)는 받지도 로드하지도 않고, DB도 열지 않는다 - OCR 경로는
 * 갤러리를 전혀 참조하지 않기 때문이다(ocr.ts는 db를 import하지 않는다).
 *
 * 이미지는 워커가 직접 받는다
 * job에는 URL만 실려 온다. pbs.twimg.com은 공개라 노트북이 자기 회선으로 받으면 되고,
 * 그래야 확장 쪽 머신의 업로드 대역폭과 브로커 용량을 둘 다 쓰지 않는다.
 *
 * 실패해도 job은 사라지지 않는다
 * 결과를 올려야만 브로커가 job을 지운다. 중간에 죽으면 lease가 만료되면서 다음 폴링에
 * 다시 잡힌다. 이미지를 한 장도 받지 못한 경우만 error를 실어 보고를 끝낸다 - 그건 재시도해도
 * 같은 결과이므로 job을 붙잡아 둘 이유가 없다.
 *
 * Env:
 *   OCR_BROKER_URL       브로커 주소. 보통 http://<브로커 머신>:3100
 *   OCR_BROKER_SECRET    요청 서명용 공유 비밀 (필수). 브로커의 OCR_BROKER_SECRET, 확장 팝업의
 *                        "공유 비밀"과 같은 값이어야 한다
 *   OCR_WORKER_INTERVAL_MS  기본 60000
 *   OCR_WORKER_ID        로그·lease 표시용
 */

import path from "path";
import nodeCrypto from "crypto";

import { ensureModelsDownloaded } from "./model-downloader";
import { initOCR, parseSearchLists, performOCR } from "./ocr";
import { loadConfig, PROJECT_ROOT } from "./config";

const MODELS_DIR = path.join(PROJECT_ROOT, "models");

const BROKER_URL = (process.env.OCR_BROKER_URL || "http://localhost:3100").replace(/\/+$/, "");
const BROKER_SECRET = process.env.OCR_BROKER_SECRET || "";
const INTERVAL_MS = Number(process.env.OCR_WORKER_INTERVAL_MS || 60000);
const WORKER_ID = process.env.OCR_WORKER_ID || `imageanalyzer-${process.pid}`;

/** 한 장 받는 데 걸리는 상한. 죽은 연결에 대한 방어이지 성능 상한이 아니다. */
const IMAGE_FETCH_TIMEOUT_MS = 60000;

/**
 * 받아올 수 있는 이미지 출처.
 *
 * 이 워커는 NAT 안쪽에 있고 job이 시키는 대로 URL을 받아온다. 즉 job을 넣을 수 있는 쪽은
 * **집 네트워크 안의 기계로 임의 주소를 fetch시킬 수 있다**(SSRF). 브로커가 뚫리거나, 잘못
 * 노출되거나, 버그로 이상한 job이 들어와도 그 지렛대를 주지 않도록 워커 자신이 막는다.
 *
 * 확장이 보내는 이미지는 x-scraper의 HTML 정규식이든 API의 media_url_https든 **항상**
 * pbs.twimg.com이므로(normalizeTwitterImageUrl 참조) 이 목록으로 잃는 것이 없다.
 * 다른 출처가 필요해지면 여기에 추가한다.
 */
const ALLOWED_IMAGE_HOSTS = new Set(["pbs.twimg.com"]);

function isAllowedImageUrl(raw: string): boolean {
  try {
    const url = new URL(raw);
    return url.protocol === "https:" && ALLOWED_IMAGE_HOSTS.has(url.hostname);
  } catch {
    return false;
  }
}

interface Job {
  jobId: string;
  postUrl: string;
  restId: string;
  imageUrls: string[];
  searchStrings: string;
  createdAt: string;
}

/**
 * 요청 서명 (스킴 OCR1)
 *
 *   Authorization: OCR1 ts=<unix ms>,nonce=<hex>,sig=<hex>
 *   서명 대상: METHOD \n path+query \n ts \n nonce \n sha256hex(body)
 *   sig      = HMAC-SHA256(secret, 위 문자열), hex
 *
 * 비밀값 자체는 회선을 지나지 않고 요청에 대한 서명만 지나간다. 정적 토큰이었다면 경로 위의
 * 관찰자가 그대로 주워 재사용할 수 있다. ts가 가로챈 요청에 만료를 주고, nonce가 그 창 안에서의
 * 재전송을 막는다.
 *
 * 응답에는 `X-Ocr-Signature = HMAC(secret, nonce \n sha256hex(body))`가 붙는다. **이 요청의**
 * nonce에 묶여 있으므로 응답 재전송이 불가능하고, 포트만 차지한 프로그램(/health handshake를
 * 위조한 경우 포함)을 첫 실호출에서 걸러낸다.
 *
 * 이 형식은 브로커(ocr-broker/server.js)와 확장(ocr-queue-client.js)에도 동일하게 있다.
 * 리포가 다른 별개 배포물이라 코드로 공유할 수 없으니 어긋나면 안 된다.
 */
function sha256Hex(text: string): string {
  return nodeCrypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function hmacHex(payload: string): string {
  return nodeCrypto.createHmac("sha256", BROKER_SECRET).update(payload, "utf8").digest("hex");
}

async function brokerFetch(pathname: string, init: RequestInit = {}): Promise<any> {
  const method = (init.method || "GET").toUpperCase();
  const bodyText = typeof init.body === "string" ? init.body : "";
  const headers: Record<string, string> = { "Content-Type": "application/json" };

  let nonce = "";
  if (BROKER_SECRET) {
    const ts = String(Date.now());
    nonce = nodeCrypto.randomBytes(12).toString("hex");
    const payload = [method, pathname, ts, nonce, sha256Hex(bodyText)].join("\n");
    headers["Authorization"] = `OCR1 ts=${ts},nonce=${nonce},sig=${hmacHex(payload)}`;
  }

  const response = await fetch(`${BROKER_URL}${pathname}`, {
    ...init,
    headers: { ...headers, ...(init.headers as Record<string, string> | undefined) },
  });

  // 텍스트로 읽어 받은 바이트 그대로 서명을 검증한다.
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}${text ? ` ${text.slice(0, 200)}` : ""}`);
  }

  // /health는 설계상 서명이 없다 (handshake이며 그것으로 아무것도 신뢰하지 않는다).
  if (BROKER_SECRET && nonce && pathname !== "/health") {
    if (response.headers.get("x-ocr-signature") !== hmacHex(`${nonce}\n${sha256Hex(text)}`)) {
      throw new Error("response signature mismatch (브로커가 보낸 응답이 아니다)");
    }
  }

  return JSON.parse(text);
}

/**
 * 기동 시 상대가 정말 브로커인지 확인한다.
 *
 * 포트를 쥐고 있는 것과 브로커인 것은 다르다. 다른 프로그램이 그 포트를 잡고 있으면 claim이
 * 매번 빈 응답처럼 보여 "할 일이 없다"와 구분되지 않고, 워커는 영원히 조용히 놀게 된다.
 *
 * 두 실패를 구분한다:
 *  - 닿지 않음 → 경고만. 브로커가 아직 안 떴을 수 있으므로 계속 폴링한다
 *  - 닿았는데 브로커가 아님 → 설정이 틀린 것이고 폴링해도 낫지 않으므로 즉시 종료한다
 */
async function verifyBroker(): Promise<void> {
  let health: any;
  try {
    health = await brokerFetch("/health");
  } catch (e: any) {
    console.warn(`Broker not reachable yet - broker: ${BROKER_URL}, error: ${e.message} (계속 폴링한다. 브로커가 아직 안 떴거나 방화벽이 막고 있을 수 있다)`);
    return;
  }

  if (health?.service !== "ocr-broker") {
    console.error(`Not the OCR broker - broker: ${BROKER_URL}, service: ${health?.service ?? "none"} (다른 프로그램이 이 포트를 잡고 있다. OCR_BROKER_URL과 브로커 포트를 확인할 것)`);
    process.exit(1);
  }

  console.log(`Broker verified - broker: ${BROKER_URL}, protocol: ${health.protocol}, jobs: ${health.jobs}, results: ${health.results}`);
}

async function claimJob(): Promise<Job | null> {
  // 한 번에 하나만 잡는다. 여러 개를 잡아 두면 앞의 것을 처리하는 동안 뒤의 것 lease가
  // 만료되어, 아무도 손대지 않은 job을 잡고 있는 척하는 구간이 생긴다.
  const data = await brokerFetch("/claim", {
    method: "POST",
    body: JSON.stringify({ max: 1, workerId: WORKER_ID }),
  });
  const jobs: Job[] = Array.isArray(data?.jobs) ? data.jobs : [];
  return jobs.length > 0 ? jobs[0] : null;
}

async function fetchImageOnce(url: string): Promise<Buffer> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), IMAGE_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 1회 재시도한다. 장당 OCR이 수십 초라 이미지 사이 간격이 그만큼 벌어지고, 그 사이 서버가
 * 유휴 keep-alive 연결을 닫으면 다음 요청이 죽은 소켓에 실려 즉사한다(실측: 12초 간격의
 * 두 번째 이미지가 그렇게 실패했다). 재연결 한 번이면 끝나는 문제다.
 *
 * 두 번 다 실패하면 그 이미지만 버린다 - 한 장 때문에 포스트 전체의 OCR을 잃지 않는다.
 * 확장의 OpenAIClient.fetchImageAsDataUrl도 같은 이유로 같은 규칙을 쓴다.
 */
async function fetchImage(url: string): Promise<Buffer | null> {
  try {
    return await fetchImageOnce(url);
  } catch (first: any) {
    const reason = first.name === "AbortError" ? "timeout" : first.message;
    console.warn(`Image fetch failed, retrying once - url: ${url}, error: ${reason}`);
  }

  try {
    return await fetchImageOnce(url);
  } catch (second: any) {
    const reason = second.name === "AbortError" ? "timeout" : second.message;
    console.warn(`Image fetch failed twice, skipping this image - url: ${url}, error: ${reason}`);
    return null;
  }
}

/**
 * 한 job의 모든 이미지를 OCR해 매칭된 줄을 합친다.
 *
 * 검색 리스트는 job이 들고 온 것이 정본이다. 서버 모드의 multipart `searchStrings`와 같은
 * 이유로, 워커가 자기 searchStrings.tsv를 보면 "목록에 없어서 못 찾음"과 "있는데 못 읽음"이
 * 구분되지 않아 확장의 미스 보고가 무의미해진다.
 */
async function runJob(job: Job): Promise<{ found: string[]; regions: number; error: string | null }> {
  const searchLists = parseSearchLists(job.searchStrings);
  if (searchLists.length === 0) {
    return { found: [], regions: 0, error: "job carried no search strings" };
  }

  const cfg = loadConfig();
  const found = new Set<string>();
  let regions = 0;
  let fetched = 0;

  for (const url of job.imageUrls) {
    if (!isAllowedImageUrl(url)) {
      // 정상 job에는 절대 나오지 않는다. 나왔다면 브로커에 이상한 것이 들어온 것이므로
      // 눈에 띄게 남긴다.
      console.warn(`Image URL rejected - jobId: ${job.jobId}, url: ${url} (허용 호스트: ${[...ALLOWED_IMAGE_HOSTS].join(", ")})`);
      continue;
    }

    const buffer = await fetchImage(url);
    if (!buffer) continue;
    fetched++;

    const started = Date.now();
    const result = await performOCR(buffer, searchLists, cfg.ocr);
    regions += result.regions;
    for (const line of result.found) found.add(line);
    console.log(
      `Image analyzed - jobId: ${job.jobId}, url: ${url}, ms: ${Date.now() - started}, regions: ${result.regions}, found: ${result.found.length}`,
    );
  }

  // 한 장도 못 받았으면 재시도해도 같으므로 error로 마감한다. 일부만 받았으면 받은 만큼의
  // 판정이 유효하므로 정상 결과로 올린다.
  const error = fetched === 0 ? `none of ${job.imageUrls.length} images could be fetched` : null;
  return { found: [...found], regions, error };
}

async function tick(): Promise<void> {
  let job: Job | null;
  try {
    job = await claimJob();
  } catch (e: any) {
    // 브로커가 꺼져 있거나 네트워크가 끊긴 상태. 다음 폴링에서 자연히 복구된다.
    console.warn(`Claim failed - broker: ${BROKER_URL}, error: ${e.message}`);
    return;
  }

  // 큐가 빈 것은 정상이므로 아무것도 찍지 않는다(1분마다 도는 루프다).
  while (job) {
    const startedAt = Date.now();
    let payload: { found: string[]; regions: number; error: string | null };
    try {
      payload = await runJob(job);
    } catch (e: any) {
      // OCR 자체가 터진 경우. 결과를 올리지 않고 lease를 놓아 다음에 다시 잡히게 한다.
      console.error(`Job failed - jobId: ${job.jobId}, error: ${e.message}`);
      try {
        await brokerFetch(`/jobs/${encodeURIComponent(job.jobId)}/release`, { method: "POST" });
      } catch (releaseErr: any) {
        console.warn(`Release failed - jobId: ${job.jobId}, error: ${releaseErr.message}`);
      }
      return;
    }

    const elapsedMs = Date.now() - startedAt;
    try {
      await brokerFetch(`/jobs/${encodeURIComponent(job.jobId)}/result`, {
        method: "POST",
        body: JSON.stringify({ ...payload, elapsedMs }),
      });
      console.log(
        `Job done - jobId: ${job.jobId}, images: ${job.imageUrls.length}, found: ${payload.found.length}, regions: ${payload.regions}, ms: ${elapsedMs}, error: ${payload.error ?? "none"}, postUrl: ${job.postUrl}`,
      );
    } catch (e: any) {
      // 결과를 못 올리면 job은 브로커에 그대로 남는다. 재시도가 곧 그 job의 재실행이다.
      console.error(`Result upload failed - jobId: ${job.jobId}, error: ${e.message}`);
      return;
    }

    // 대기 중인 job이 더 있으면 다음 폴링을 기다리지 않고 이어서 비운다. 노트북이 오래
    // 꺼져 있다가 켜졌을 때 1분에 하나씩 처리하면 하루치를 따라잡지 못한다.
    try {
      job = await claimJob();
    } catch (e: any) {
      console.warn(`Claim failed - broker: ${BROKER_URL}, error: ${e.message}`);
      return;
    }
  }
}

async function main() {
  console.log(`Initializing OCR worker - broker: ${BROKER_URL}, intervalMs: ${INTERVAL_MS}, workerId: ${WORKER_ID}, auth: ${BROKER_SECRET ? "OCR1 signed" : "none"}`);

  // 모델 로드 전에 확인한다. 주소가 틀렸으면 ONNX를 몇 초 들여 올릴 이유가 없다.
  await verifyBroker();

  // OCR만 쓰므로 비전 모델은 받지도 로드하지도 않는다(ocrOnly=true).
  await ensureModelsDownloaded(MODELS_DIR, true);
  await initOCR(MODELS_DIR);
  console.log("OCR worker ready");

  // 겹치지 않게 한 틱이 끝난 뒤에 다음 간격을 잰다. OCR은 장당 수십 초라 setInterval이면
  // 앞 틱이 끝나기 전에 다음 틱이 겹칠 수 있다.
  for (;;) {
    try {
      await tick();
    } catch (e: any) {
      console.error(`Tick failed - error: ${e.message}`);
    }
    await new Promise((resolve) => setTimeout(resolve, INTERVAL_MS));
  }
}

main().catch((err) => {
  console.error("Worker startup failed:", err);
  process.exit(1);
});
