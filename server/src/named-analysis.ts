/** Match requested data-folder names using the existing gallery and confidence rules. */
import fs from "fs";
import path from "path";
import { analyzeImage, Config, Match } from "./analyze";
import { getGallery, Kind, Space } from "./db";
import { PROJECT_ROOT } from "./config";

const GALLERIES: [Kind, Space, string][] = [
  ["face", "arcface", "faces"], ["character", "ccip", "characters"], ["costume", "wdtag", "costumes"],
];

export function availableNames(requested: string[]): string[] {
  const registered = new Set<string>();
  for (const [kind, space, folder] of GALLERIES) {
    const root = path.join(PROJECT_ROOT, "data", folder);
    const folders = new Set(fs.existsSync(root) ? fs.readdirSync(root, { withFileTypes: true })
      .filter(entry => entry.isDirectory()).map(entry => entry.name) : []);
    for (const item of getGallery(kind, space)) if (folders.has(item.name)) registered.add(item.name);
  }
  return [...new Set(requested)].filter(name => registered.has(name));
}

export async function analyzeNamedImage(buffer: Buffer, names: string[], cfg: Config): Promise<(Match & { kind: string })[]> {
  if (!names.length) return [];
  // Retain all gallery competitors for margin checks; filter only the confirmed output.
  // Empty OCR targets skip OCR: its independent job remains the sole OCR execution.
  const result = await analyzeImage(buffer, [], cfg);
  return (["faces", "characters", "costumes"] as const).flatMap(kind =>
    result[kind].filter(match => names.includes(match.name)).map(match => ({ ...match, kind })));
}
