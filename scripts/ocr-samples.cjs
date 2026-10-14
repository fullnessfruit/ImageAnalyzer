/** Run the real OCR models against locally labeled keyword-presence images. */
const fs = require("fs");
const path = require("path");
const { initOCR, parseSearchLists, performOCR } = require("../server/dist/ocr.js");
const { loadConfig } = require("../server/dist/config.js");

const root = path.resolve(__dirname, "..");
const sampleDir = path.join(root, "sample");
const extensions = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

async function main() {
  await initOCR(path.join(root, "models"));
  const ocrConfig = loadConfig().ocr;
  let checked = 0;
  let failures = 0;

  async function check(file, keyword, expected) {
    checked++;
    try {
      const result = await performOCR(fs.readFileSync(file), parseSearchLists(keyword), ocrConfig);
      const found = result.found.includes(keyword);
      const score = result.detail[0]?.parts[0]?.score ?? 0;
      if (found !== expected) failures++;
      console.log(`OCR sample - file: ${JSON.stringify(path.relative(root, file))}, keyword: ${JSON.stringify(keyword)}, expected: ${expected}, found: ${found}, score: ${score.toFixed(4)}, ok: ${found === expected}`);
    } catch (error) {
      failures++;
      console.error(`OCR sample failed - file: ${JSON.stringify(path.relative(root, file))}, keyword: ${JSON.stringify(keyword)}, error: ${error.message}`);
    }
  }

  for (const [folder, expected] of [["keyword-present", true], ["keyword-absent", false]]) {
    const dir = path.join(sampleDir, folder);
    if (!fs.existsSync(dir)) throw new Error(`OCR sample folder missing: ${dir}`);
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const keyword = entry.name;
      const keywordDir = path.join(dir, keyword);
      for (const file of fs.readdirSync(keywordDir).sort()) {
        if (!extensions.has(path.extname(file).toLowerCase())) continue;
        await check(path.join(keywordDir, file), keyword, expected);
      }
    }
  }

  // These negative-for-あぐり images also contain real words that must stay detectable.
  for (const [file, keyword] of [
    ["HRWKnRBW4AUg9iL.jpg", "アクリル"],
    ["HRf_zQybUAAnIqU.png", "めぐり"],
  ]) {
    const image = path.join(sampleDir, "keyword-absent", "あぐり", file);
    if (fs.existsSync(image)) await check(image, keyword, true);
  }

  if (checked === 0) throw new Error("No OCR sample images found");
  console.log(`OCR sample summary - checked: ${checked}, failures: ${failures}`);
  if (failures > 0) process.exitCode = 1;
}

main().catch(error => { console.error(`OCR samples failed - error: ${error.message}`); process.exitCode = 1; });
