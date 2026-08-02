/**
 * searchStrings.tsv 관리 CLI.
 *
 * 한 줄이 하나의 리스트다. 줄 안에서 탭으로 나눈 파트는 모두 존재해야 하고(AND),
 * 줄끼리는 OR라 한 리스트라도 일치하면 매칭 성공이다.
 *
 * Usage:
 *   npx ts-node scripts/search-strings.ts --add "大西亜玖璃"
 *   npx ts-node scripts/search-strings.ts --add "上原歩夢<TAB>虹ヶ咲"
 *   npx ts-node scripts/search-strings.ts --remove "大西亜玖璃"
 *   npx ts-node scripts/search-strings.ts --list
 */

import fs from "fs";
import path from "path";

const TSV_PATH = path.resolve(__dirname, "..", "searchStrings.tsv");

function loadLists(): string[] {
  if (!fs.existsSync(TSV_PATH)) return [];
  return fs
    .readFileSync(TSV_PATH, "utf-8")
    .split("\n")
    .map((line) => line.replace(/\r$/, "").trimEnd())
    .filter((line) => line.length > 0);
}

function saveLists(lists: string[]): void {
  fs.writeFileSync(TSV_PATH, lists.join("\n") + "\n", "utf-8");
}

function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes("--help")) {
    console.log("Usage:");
    console.log('  npx ts-node scripts/search-strings.ts --add "문자열"      (탭으로 복합 조건)');
    console.log('  npx ts-node scripts/search-strings.ts --remove "문자열"');
    console.log("  npx ts-node scripts/search-strings.ts --list");
    return;
  }

  const lists = loadLists();

  if (args.includes("--list")) {
    console.log(`searchStrings.tsv (${lists.length} lists):`);
    lists.forEach((s, i) => console.log(`  ${i + 1}. ${s.split("\t").join("  AND  ")}`));
    return;
  }

  const addIdx = args.indexOf("--add");
  if (addIdx !== -1 && args[addIdx + 1]) {
    const value = args[addIdx + 1];
    if (lists.includes(value)) {
      console.log(`이미 존재한다: "${value}"`);
    } else {
      lists.push(value);
      saveLists(lists);
      console.log(`추가됨: "${value}" (총 ${lists.length}개 리스트)`);
    }
    return;
  }

  const removeIdx = args.indexOf("--remove");
  if (removeIdx !== -1 && args[removeIdx + 1]) {
    const value = args[removeIdx + 1];
    const idx = lists.indexOf(value);
    if (idx === -1) {
      console.log(`찾을 수 없다: "${value}"`);
    } else {
      lists.splice(idx, 1);
      saveLists(lists);
      console.log(`삭제됨: "${value}" (총 ${lists.length}개 리스트)`);
    }
    return;
  }

  console.log("Invalid arguments. Use --help for usage.");
}

main();
