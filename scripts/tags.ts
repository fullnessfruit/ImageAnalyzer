/**
 * WD-Tagger 원시 출력 확인 도구.
 * 캐릭터 임계값 조정, 의상 태그 확인, 태거가 무엇을 보고 있는지 진단할 때 쓴다.
 *
 * Usage: npx ts-node scripts/tags.ts <이미지경로>
 */

import path from "path";
import fs from "fs";
import { loadModels, taggerProbs } from "../server/src/inference";

const PROJECT_ROOT = path.resolve(__dirname, "..");

async function main() {
  const target = process.argv[2];
  if (!target) {
    console.error("Usage: npx ts-node scripts/tags.ts <이미지경로>");
    process.exit(1);
  }

  await loadModels(path.join(PROJECT_ROOT, "models"));
  const result = await taggerProbs(fs.readFileSync(path.resolve(PROJECT_ROOT, target)));
  if (!result) {
    console.error("Tagger unavailable");
    process.exit(1);
  }

  const { probs, tags } = result;
  const rows = tags.map((t, i) => ({ ...t, prob: i < probs.length ? probs[i] : 0 }));

  for (const [label, category] of [["RATING", 9], ["CHARACTER", 4], ["GENERAL", 0]] as const) {
    const top = rows.filter((r) => r.category === category).sort((a, b) => b.prob - a.prob).slice(0, 20);
    console.log(`\n--- ${label} (top 20)`);
    for (const r of top) console.log(`  ${r.prob.toFixed(4)}  ${r.name}`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
