/**
 * CLI tool to add/remove search strings in searchStrings.tsv.
 *
 * Usage:
 *   npx ts-node scripts/update-config.ts --add "string"
 *   npx ts-node scripts/update-config.ts --remove "string"
 *   npx ts-node scripts/update-config.ts --list
 */

import fs from "fs";
import path from "path";

const TSV_PATH = path.resolve(__dirname, "..", "searchStrings.tsv");

function loadStrings(): string[] {
  if (!fs.existsSync(TSV_PATH)) return [];
  return fs
    .readFileSync(TSV_PATH, "utf-8")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0);
}

function saveStrings(strings: string[]): void {
  fs.writeFileSync(TSV_PATH, strings.join("\n") + "\n", "utf-8");
}

function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes("--help")) {
    console.log("Usage:");
    console.log('  npx ts-node scripts/update-config.ts --add "string"');
    console.log('  npx ts-node scripts/update-config.ts --remove "string"');
    console.log("  npx ts-node scripts/update-config.ts --list");
    return;
  }

  const strings = loadStrings();

  if (args.includes("--list")) {
    console.log(`searchStrings.tsv (${strings.length} entries):`);
    strings.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
    return;
  }

  const addIdx = args.indexOf("--add");
  if (addIdx !== -1 && args[addIdx + 1]) {
    const value = args[addIdx + 1];
    if (strings.includes(value)) {
      console.log(`"${value}" already exists.`);
    } else {
      strings.push(value);
      saveStrings(strings);
      console.log(`Added "${value}". Total: ${strings.length} entries.`);
    }
    return;
  }

  const removeIdx = args.indexOf("--remove");
  if (removeIdx !== -1 && args[removeIdx + 1]) {
    const value = args[removeIdx + 1];
    const idx = strings.indexOf(value);
    if (idx === -1) {
      console.log(`"${value}" not found.`);
    } else {
      strings.splice(idx, 1);
      saveStrings(strings);
      console.log(`Removed "${value}". Total: ${strings.length} entries.`);
    }
    return;
  }

  console.log("Invalid arguments. Use --help for usage.");
}

main();
