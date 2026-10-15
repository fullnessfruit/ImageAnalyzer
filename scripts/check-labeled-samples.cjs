/** Recheck user-labeled local images through the same analysis path as the worker. */
const fs = require("fs");
const path = require("path");
const { initDB } = require("../server/dist/db.js");
const { loadModels } = require("../server/dist/inference.js");
const { initOCR, parseSearchLists } = require("../server/dist/ocr.js");
const { analyzeImage } = require("../server/dist/analyze.js");
const { loadConfig } = require("../server/dist/config.js");

const root = path.resolve(__dirname, "..");
const labels = JSON.parse(fs.readFileSync(path.join(root, "sample-labels.json"), "utf8"));

function byFile(groups) {
  const files = new Map();
  for (const [name, names] of Object.entries(groups)) {
    for (const file of names) {
      if (!files.has(file)) files.set(file, []);
      files.get(file).push(name);
    }
  }
  return files;
}

async function main() {
  initDB(path.join(root, "db"));
  await loadModels(path.join(root, "models"));
  await initOCR(path.join(root, "models"));

  const faces = byFile(labels.faces);
  const unregisteredFaces = byFile(labels.unregisteredFaces);
  const bodyOnly = byFile(labels.bodyOnly);
  const ocr = byFile(labels.ocr);
  const optionalOcr = byFile(labels.optionalOcr);
  const requested = new Set(process.argv.slice(2));
  const files = [...new Set([...faces.keys(), ...unregisteredFaces.keys(), ...bodyOnly.keys(), ...ocr.keys(), ...optionalOcr.keys()])]
    .filter(file => !requested.size || requested.has(file)).sort();
  if (!files.length || (requested.size && files.length !== requested.size)) throw new Error("Unknown labeled sample filename");
  const cfg = loadConfig();
  cfg.candidates = { ...cfg.candidates, enabled: false };

  let failed = 0;
  for (const file of files) {
    const imagePath = path.join(root, "sample", file);
    const expectedFaces = faces.get(file) || [];
    const expectedOcr = ocr.get(file) || [];
    const optional = optionalOcr.get(file) || [];
    const terms = [...new Set([...expectedOcr, ...optional])];
    const errors = [];
    try {
      const image = fs.readFileSync(imagePath);
      const result = await analyzeImage(image, terms.length ? parseSearchLists(terms.join("\n")) : [], cfg);
      const foundFaces = [...new Set(result.faces.map(face => face.name))];
      if (!bodyOnly.has(file)) {
        for (const name of expectedFaces) if (!foundFaces.includes(name)) errors.push(`face missing: ${name}`);
        for (const name of foundFaces) if (!expectedFaces.includes(name)) errors.push(`unexpected face: ${name}`);
      }
      for (const term of expectedOcr) if (!result.ocr.found.includes(term)) errors.push(`OCR missing: ${term}`);
      console.log(`Labeled sample - ${JSON.stringify({ file, expectedFaces, unregisteredFaces: unregisteredFaces.get(file) || [], bodyOnly: bodyOnly.get(file) || [], faces: result.faces, facesWeak: result.facesWeak, ocr: terms.map(term => ({ term, required: expectedOcr.includes(term), found: result.ocr.found.includes(term), score: result.ocr.detail.find(item => item.list === term)?.parts[0]?.score ?? 0 })), detections: result._detections, elapsedMs: result._elapsedMs, errors })}`);
    } catch (error) {
      errors.push(error.message);
      console.error(`Labeled sample failed - ${JSON.stringify({ file, errors })}`);
    }
    if (errors.length) failed++;
  }
  console.log(`Labeled sample summary - images: ${files.length}, failed: ${failed}`);
  if (failed) process.exitCode = 1;
}

main().catch(error => { console.error(`Labeled samples failed - error: ${error.message}`); process.exitCode = 1; });
