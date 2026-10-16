/** Validate the human labels before expensive model inference. */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const sampleDir = path.join(root, "sample");
const imageExtension = /\.(?:jpe?g|png|webp|bmp)$/i;

function imageFiles(dir, prefix = "") {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(item => {
    const relative = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isDirectory()) return imageFiles(path.join(dir, item.name), relative);
    return item.isFile() && imageExtension.test(item.name) ? [relative] : [];
  });
}

function uniqueNames(value, context) {
  if (!Array.isArray(value) || value.some(name => typeof name !== "string" || !name)) {
    throw new Error(`${context} must be a list of nonempty names`);
  }
  if (new Set(value).size !== value.length) throw new Error(`${context} contains duplicate names`);
  return value;
}

function loadSampleLabels() {
  const labels = JSON.parse(fs.readFileSync(path.join(sampleDir, "labels.json"), "utf8"));
  if (labels.format !== "imageanalyzer-sample-labels-v2") throw new Error("Unknown sample label format");
  const people = uniqueNames(labels.subjectCatalog?.people, "subjectCatalog.people");
  const characters = uniqueNames(labels.subjectCatalog?.characters, "subjectCatalog.characters");
  const knownPeople = new Set(people);
  const knownCharacters = new Set(characters);
  if (!labels.images || typeof labels.images !== "object" || Array.isArray(labels.images)) throw new Error("images must be an object");
  const seenPeople = new Set();
  const seenCharacters = new Set();
  const seenOcr = new Set();
  for (const [file, entry] of Object.entries(labels.images)) {
    if (!imageExtension.test(file) || path.isAbsolute(file) || file.split("/").includes("..")) throw new Error(`Invalid sample path: ${file}`);
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error(`Invalid label entry: ${file}`);
    const entryPeople = uniqueNames(entry.people || [], `${file}.people`);
    const entryCharacters = uniqueNames(entry.characters || [], `${file}.characters`);
    for (const name of entryPeople) {
      if (!knownPeople.has(name)) throw new Error(`Uncataloged person: ${file} ${name}`);
      seenPeople.add(name);
    }
    for (const name of entryCharacters) {
      if (!knownCharacters.has(name)) throw new Error(`Uncataloged character: ${file} ${name}`);
      seenCharacters.add(name);
    }
    for (const name of uniqueNames(entry.faceChecks || [], `${file}.faceChecks`)) {
      if (!entryPeople.includes(name)) throw new Error(`Face check without person label: ${file} ${name}`);
    }
    for (const term of uniqueNames(entry.ocr || [], `${file}.ocr`)) seenOcr.add(term);
    for (const term of uniqueNames(entry.optionalOcr || [], `${file}.optionalOcr`)) seenOcr.add(term);
    for (const term of uniqueNames(entry.absentOcr || [], `${file}.absentOcr`)) {
      if ([...(entry.ocr || []), ...(entry.optionalOcr || [])].includes(term)) throw new Error(`OCR term both present and absent: ${file} ${term}`);
    }
    if (entry.ocrText) for (const term of Object.keys(entry.ocrText)) {
      if (![...(entry.ocr || []), ...(entry.optionalOcr || [])].includes(term)) throw new Error(`OCR text without query: ${file} ${term}`);
    }
    if (entry.subjectNotes) for (const name of Object.keys(entry.subjectNotes)) {
      if (![...entryPeople, ...entryCharacters].includes(name)) throw new Error(`Note without subject: ${file} ${name}`);
    }
    if (entry.imageNotes !== undefined && (typeof entry.imageNotes !== "string" || !entry.imageNotes.trim())) throw new Error(`Invalid image notes: ${file}`);
    if (entry.faceCheck && entry.faceCheck !== "skip") throw new Error(`Unknown face check mode: ${file}`);
    if (entry.absent) {
      if (entry.absent !== "allCatalogedSubjects" || entryPeople.length || entryCharacters.length) throw new Error(`Invalid absence label: ${file}`);
    }
  }
  if (people.some(name => !seenPeople.has(name) && !seenOcr.has(name)) || characters.some(name => !seenCharacters.has(name) && !seenOcr.has(name))) {
    throw new Error("Subject catalog contains a name with no image or OCR label");
  }
  const unlabeled = uniqueNames(labels.unlabeledImages, "unlabeledImages");
  const listed = [...Object.keys(labels.images), ...unlabeled];
  if (new Set(listed).size !== listed.length) throw new Error("An image is both labeled and unlabeled");
  const actual = imageFiles(sampleDir).sort();
  const expected = listed.sort();
  if (actual.length !== expected.length || actual.some((file, i) => file !== expected[i])) {
    throw new Error(`Sample inventory differs from labels.json - missing: ${expected.filter(file => !actual.includes(file)).join(",")}, unlisted: ${actual.filter(file => !expected.includes(file)).join(",")}`);
  }
  for (const file of Object.keys(labels.trainingMedia || {})) {
    if (!fs.existsSync(path.join(root, file))) throw new Error(`Training media missing: ${file}`);
  }
  return labels;
}

if (require.main === module) {
  try {
    const labels = loadSampleLabels();
    console.log(`Sample labels valid - ${JSON.stringify({ labeled: Object.keys(labels.images).length, unlabeled: labels.unlabeledImages.length, people: labels.subjectCatalog.people.length, characters: labels.subjectCatalog.characters.length })}`);
  } catch (error) {
    console.error(`Sample labels invalid - error: ${error.message}`);
    process.exitCode = 1;
  }
}

module.exports = { loadSampleLabels };
