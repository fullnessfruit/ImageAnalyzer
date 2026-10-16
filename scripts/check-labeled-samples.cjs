/** Recheck user-labeled local images through the same analysis path as the worker. */
const fs = require("fs");
const path = require("path");
const { initDB } = require("../server/dist/db.js");
const { loadModels } = require("../server/dist/inference.js");
const { initOCR, parseSearchLists } = require("../server/dist/ocr.js");
const { analyzeImage } = require("../server/dist/analyze.js");
const { loadConfig } = require("../server/dist/config.js");
const { loadSampleLabels } = require("./sample-labels.cjs");

const root = path.resolve(__dirname, "..");

async function main() {
  const labels = loadSampleLabels();
  const requested = new Set(process.argv.slice(2));
  const files = Object.keys(labels.images).filter(file => !requested.size || requested.has(file)).sort();
  if (!files.length || (requested.size && files.length !== requested.size)) throw new Error("Unknown labeled sample filename");

  initDB(path.join(root, "db"));
  await loadModels(path.join(root, "models"));
  await initOCR(path.join(root, "models"));

  const cfg = loadConfig();
  cfg.candidates = { ...cfg.candidates, enabled: false };
  const absentPeople = new Set(labels.subjectCatalog.people);
  const absentCharacters = new Set(labels.subjectCatalog.characters);

  let failed = 0;
  for (const file of files) {
    const label = labels.images[file];
    const imagePath = path.join(root, "sample", file);
    const expectedFaces = label.faceChecks || [];
    const expectedCharacters = label.characters || [];
    const expectedOcr = label.ocr || [];
    const optional = label.optionalOcr || [];
    const absentOcr = label.absentOcr || [];
    const terms = [...new Set([...expectedOcr, ...optional, ...absentOcr])];
    const errors = [];
    try {
      const image = fs.readFileSync(imagePath);
      const result = await analyzeImage(image, terms.length ? parseSearchLists(terms.join("\n")) : [], cfg);
      const foundFaces = [...new Set(result.faces.map(face => face.name))];
      const foundCharacters = [...new Set(result.characters.map(character => character.name))];
      if (label.absent === "allCatalogedSubjects") {
        for (const name of foundFaces) if (absentPeople.has(name)) errors.push(`unexpected face: ${name}`);
        for (const name of foundCharacters) if (absentCharacters.has(name)) errors.push(`unexpected character: ${name}`);
      } else if (label.faceCheck !== "skip" && expectedFaces.length) {
        for (const name of expectedFaces) if (!foundFaces.includes(name)) errors.push(`face missing: ${name}`);
        for (const name of foundFaces) if (!expectedFaces.includes(name)) errors.push(`unexpected face: ${name}`);
      }
      for (const term of expectedOcr) if (!result.ocr.found.includes(term)) errors.push(`OCR missing: ${term}`);
      for (const term of absentOcr) if (result.ocr.found.includes(term)) errors.push(`unexpected OCR: ${term}`);
      console.log(`Labeled sample - ${JSON.stringify({ file, label, faces: result.faces, facesWeak: result.facesWeak, characters: result.characters, charactersWeak: result.charactersWeak, missingLabeledCharacters: expectedCharacters.filter(name => !foundCharacters.includes(name)), ocr: terms.map(term => ({ term, required: expectedOcr.includes(term), forbidden: absentOcr.includes(term), found: result.ocr.found.includes(term), score: result.ocr.detail.find(item => item.list === term)?.parts[0]?.score ?? 0 })), detections: result._detections, elapsedMs: result._elapsedMs, errors })}`);
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
