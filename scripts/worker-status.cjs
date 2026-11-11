/** Show the worker's actual audit trail even when Task Scheduler hides its console. */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const logDirectory = path.join(__dirname, '..', 'log');
const logName = /^analysis-log_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_(\d+)\.log$/;

function processAlive(pid) {
  if (process.platform !== 'win32') return null;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === 'EPERM' ? true : false;
  }
}

function latestWorkerLog() {
  if (!fs.existsSync(logDirectory)) return null;
  const files = fs.readdirSync(logDirectory).flatMap(name => {
    const match = logName.exec(name);
    if (!match) return [];
    const file = path.join(logDirectory, name);
    return [{ file, pid: Number(match[1]), modifiedAt: fs.statSync(file).mtimeMs }];
  }).sort((a, b) => b.modifiedAt - a.modifiedAt);
  return files.find(item => processAlive(item.pid) === true) || files[0] || null;
}

function lastEntry(file) {
  const handle = fs.openSync(file, 'r');
  try {
    const size = fs.fstatSync(handle).size;
    for (let length = Math.min(64 * 1024, size); length <= size; length = Math.min(length * 2, size)) {
      const start = size - length;
      const buffer = Buffer.alloc(length);
      fs.readSync(handle, buffer, 0, length, start);
      let text = buffer.toString('utf8');
      if (start > 0) text = text.slice(text.indexOf('\n') + 1);
      const lines = text.trimEnd().split('\n');
      for (let i = lines.length - 1; i >= 0; i--) {
        const match = /^\[(\d{4}-\d{2}-\d{2}T[^\]]+Z)\] (\{.*\})$/.exec(lines[i]);
        if (!match) continue;
        try { return { time: match[1], entry: JSON.parse(match[2]) }; } catch { /* partial append */ }
      }
      if (start === 0) return null;
    }
    return null;
  } finally {
    fs.closeSync(handle);
  }
}

function describe(entry) {
  if (entry.event === 'image_request') return '이미지 분석 중';
  if (entry.event === 'image_result') {
    const states = { analyzed: '완료', cached: '캐시 재사용', reused: '동일 이미지 재사용',
      fetch_failed: '다운로드 실패', analysis_failed: '분석 실패', ocr_failed: 'OCR 실패',
      no_search_strings: '검색어 없음', rejected_url: '주소 거부', no_registered_names: '등록 이름 없음' };
    return `이미지 판정 ${states[entry.status] || entry.status || '완료'}`;
  }
  if (entry.event === 'job_result') return '결과 업로드 중';
  if (entry.event === 'job_delivery') {
    const states = { uploaded: '결과 업로드 완료', already_completed: '중복 결과 처리됨', upload_failed: '결과 업로드 실패' };
    return states[entry.status] || `결과 업로드 ${entry.status}`;
  }
  if (entry.event === 'job_error') return '작업 오류';
  if (entry.event === 'broker_poll') {
    const states = { idle: '브로커 연결 정상, 대기 중', connected: '브로커 연결됨',
      recovered: '브로커 재연결됨', failed: '브로커 요청 실패' };
    return states[entry.status] || `브로커 ${entry.status}`;
  }
  return entry.event || '알 수 없음';
}

function printStatus() {
  const now = new Date();
  const log = latestWorkerLog();
  if (!log) {
    console.log(`[${now.toISOString()}] 워커 상태 - 분석 로그가 없습니다`);
    return;
  }
  const event = lastEntry(log.file);
  const alive = processAlive(log.pid);
  const processStatus = alive === true ? '실행 중' : alive === false ? '종료됨' : '확인 불가';
  if (!event) {
    console.log(`[${now.toISOString()}] 워커 상태 - PID: ${log.pid}, 프로세스: ${processStatus}, 로그: ${path.basename(log.file)}, 기록 없음`);
    return;
  }
  const ageSec = Math.max(0, Math.floor((now.getTime() - Date.parse(event.time)) / 1000));
  const job = event.entry.jobId ? `, 작업: ${event.entry.jobId}` : '';
  const post = event.entry.postUrl ? `, 게시물: ${event.entry.postUrl}` : '';
  const error = event.entry.error ? `, 오류: ${String(event.entry.error).replace(/\s+/g, ' ').slice(0, 180)}` : '';
  const uncertainty = ageSec >= 600 ? ', 최근 기록 없음: 폴링 여부는 확인 필요' : '';
  console.log(`[${now.toISOString()}] 워커 상태 - PID: ${log.pid}, 프로세스: ${processStatus}, 마지막 기록: ${describe(event.entry)} (${event.time}, ${ageSec}초 전)${job}${post}${error}${uncertainty}`);
}

printStatus();
if (process.argv.includes('--watch')) setInterval(printStatus, 10 * 1000);
