/**
 * 검출 결과 확인 도구. 박스 좌표가 이미지 범위 안에 있는지, 얼굴 랜드마크가
 * 제대로 나오는지 진단한다.
 *
 * Usage: npx ts-node scripts/detect.ts <이미지경로>
 */

import path from "path";
import fs from "fs";
import sharp from "sharp";
import { loadModels, detectFaces, detectAnimeFaces, detectAnimePersons } from "../server/src/inference";

const PROJECT_ROOT = path.resolve(__dirname, "..");

async function main() {
  const target = process.argv[2];
  if (!target) {
    console.error("Usage: npx ts-node scripts/detect.ts <이미지경로>");
    process.exit(1);
  }

  await loadModels(path.join(PROJECT_ROOT, "models"));
  const buf = fs.readFileSync(path.resolve(PROJECT_ROOT, target));
  const meta = await sharp(buf).metadata();
  console.log(`image: ${meta.width}x${meta.height}`);

  const [real, animeFace, animePerson] = await Promise.all([
    detectFaces(buf),
    detectAnimeFaces(buf),
    detectAnimePersons(buf),
  ]);

  const show = (label: string, dets: typeof real) => {
    console.log(`\n--- ${label} (${dets.length})`);
    for (const d of dets) {
      const [x, y, w, h] = d.box.map(Math.round);
      const lm = d.landmarks ? ` lm=[${d.landmarks.map((v) => Math.round(v)).join(",")}]` : "";
      console.log(`  conf=${d.confidence.toFixed(3)} box=[${x},${y},${w},${h}]${lm}`);
    }
  };

  show("real faces (SCRFD)", real);
  show("anime faces", animeFace);
  show("anime persons", animePerson);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
