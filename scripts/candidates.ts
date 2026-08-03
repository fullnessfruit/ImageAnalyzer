/**
 * 후보 관리 - 분석 중 모아둔 크롭을 검토하고 갤러리로 넘긴다.
 *
 * 갤러리에 자동으로 들어가는 것은 없다. 여기서 추린 것만 data/faces|characters|costumes/ 로
 * 옮겨지고, 그 다음 register를 돌려야 실제 등록된다.
 *
 * Usage:
 *   npx ts-node scripts/candidates.ts --list [kind] [name]
 *   npx ts-node scripts/candidates.ts --promote <kind> <name>   # data/<kind>/<name>/ 로 이동
 *   npx ts-node scripts/candidates.ts --clear [kind] [name]     # 후보 삭제 (갤러리는 그대로)
 */

import path from "path";
import fs from "fs";
import { initDB, listCandidates, deleteCandidates, deleteCandidateByPath, Kind } from "../server/src/db";
import { CANDIDATES_DIR } from "../server/src/candidates";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(PROJECT_ROOT, "data");

/** kind → data/ 하위 등록 폴더명. register.ts의 배치와 같아야 한다. */
const TARGET_DIR: Record<string, string> = {
  face: "faces",
  character: "characters",
  costume: "costumes",
};

function list(kind?: Kind, name?: string): void {
  const rows = listCandidates(kind, name);
  if (rows.length === 0) {
    console.log("후보 없음.");
    return;
  }

  const byGroup = new Map<string, typeof rows>();
  for (const r of rows) {
    const key = `${r.kind}/${r.name}`;
    const list = byGroup.get(key);
    if (list) list.push(r);
    else byGroup.set(key, [r]);
  }

  for (const [key, group] of byGroup) {
    console.log(`\n${key} - ${group.length}건`);
    for (const r of group) {
      const exists = fs.existsSync(path.join(DATA_DIR, r.file_path)) ? "" : "  [파일 없음]";
      console.log(`  ${r.score.toFixed(4)}  ${r.source.padEnd(8)}  ${r.file_path}${exists}`);
    }
  }
  console.log(`\n총 ${rows.length}건. 검토 후 --promote 하거나 파일을 직접 옮겨라.`);
}

function promote(kind: Kind, name: string): void {
  const target = TARGET_DIR[kind];
  if (!target) {
    console.error(`Unknown kind: ${kind}. Use one of: ${Object.keys(TARGET_DIR).join(", ")}`);
    process.exit(1);
  }

  const rows = listCandidates(kind, name);
  if (rows.length === 0) {
    console.log(`후보 없음 - kind: ${kind}, name: ${name}`);
    return;
  }

  const destDir = path.join(DATA_DIR, target, name);
  fs.mkdirSync(destDir, { recursive: true });

  let moved = 0;
  let missing = 0;
  for (const r of rows) {
    const src = path.join(DATA_DIR, r.file_path);
    if (!fs.existsSync(src)) {
      deleteCandidateByPath(r.file_path);
      missing++;
      continue;
    }
    const dest = path.join(destDir, path.basename(src));
    fs.renameSync(src, dest);
    deleteCandidateByPath(r.file_path);
    moved++;
  }

  console.log(`Promoted - kind: ${kind}, name: ${name}, moved: ${moved}, missing: ${missing}, dest: ${path.relative(PROJECT_ROOT, destDir)}`);
  console.log(`   등록하려면: npm run register:${kind === "face" ? "faces" : kind === "character" ? "characters" : "costumes"}`);
}

function clear(kind?: Kind, name?: string): void {
  const rows = listCandidates(kind, name);
  for (const r of rows) {
    const p = path.join(DATA_DIR, r.file_path);
    if (fs.existsSync(p)) fs.rmSync(p);
  }
  const removed = deleteCandidates(kind, name);
  console.log(`Cleared - kind: ${kind ?? "all"}, name: ${name ?? "all"}, removed: ${removed}`);
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes("--help")) {
    console.log("Usage:");
    console.log("  npx ts-node scripts/candidates.ts --list [kind] [name]");
    console.log("  npx ts-node scripts/candidates.ts --promote <kind> <name>");
    console.log("  npx ts-node scripts/candidates.ts --clear [kind] [name]");
    console.log(`  kind: ${Object.keys(TARGET_DIR).join(" | ")}`);
    return;
  }

  initDB(path.join(PROJECT_ROOT, "db"));

  const cmd = args[0];
  const kind = args[1] as Kind | undefined;
  const name = args[2];

  if (cmd === "--list") list(kind, name);
  else if (cmd === "--promote") {
    if (!kind || !name) {
      console.error("--promote 는 kind 와 name 이 모두 필요하다.");
      process.exit(1);
    }
    promote(kind, name);
  } else if (cmd === "--clear") clear(kind, name);
  else console.log("Invalid arguments. Use --help for usage.");
}

main();
