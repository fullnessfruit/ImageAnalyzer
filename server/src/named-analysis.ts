/** Match requested data-folder names using the existing gallery and confidence rules. */
import fs from "fs";
import path from "path";
import { analyzeImage, AnalyzeResult, Config, Match } from "./analyze";
import { getGallery, Kind, Space } from "./db";
import { PROJECT_ROOT } from "./config";

const GALLERIES: [Kind, Space, string][] = [
  ["face", "arcface", "faces"], ["character", "ccip", "characters"], ["costume", "wdtag", "costumes"],
];

interface RequestedTarget {
  name: string;
  kind?: Kind;
}

function resolveTarget(raw: string, registered: ReadonlyMap<string, ReadonlySet<Kind>>): RequestedTarget | null {
  // An exact folder name takes precedence, including names containing a colon.
  if (registered.has(raw)) return { name: raw };
  const match = /^(face|character|costume):(.+)$/.exec(raw);
  if (!match) return null;
  const kind = match[1] as Kind;
  const name = match[2];
  return registered.get(name)?.has(kind) ? { name, kind } : null;
}

export function registeredKinds(): Map<string, Set<Kind>> {
  const registered = new Map<string, Set<Kind>>();
  for (const [kind, space, folder] of GALLERIES) {
    const root = path.join(PROJECT_ROOT, "data", folder);
    const folders = new Set(fs.existsSync(root) ? fs.readdirSync(root, { withFileTypes: true })
      .filter(entry => entry.isDirectory()).map(entry => entry.name) : []);
    for (const item of getGallery(kind, space)) {
      if (!folders.has(item.name)) continue;
      const kinds = registered.get(item.name) || new Set<Kind>();
      kinds.add(kind);
      registered.set(item.name, kinds);
    }
  }
  return registered;
}

export function availableNames(requested: string[]): string[] {
  const registered = registeredKinds();
  return [...new Set(requested)].filter(name => resolveTarget(name, registered) !== null);
}

export function selectNamedMatches(
  result: Pick<AnalyzeResult, "faces" | "characters" | "costumes">,
  names: string[],
  registered: ReadonlyMap<string, ReadonlySet<Kind>>,
): (Match & { kind: string })[] {
  const requested = names.map(name => resolveTarget(name, registered)).filter((target): target is RequestedTarget => target !== null);
  return ([
    ["faces", "face"], ["characters", "character"], ["costumes", "costume"],
  ] as const).flatMap(([field, kind]) => result[field]
    .filter(match => registered.get(match.name)?.has(kind) &&
      requested.some(target => target.name === match.name && (!target.kind || target.kind === kind)))
    .map(match => ({ ...match, kind: field })));
}

export function foundNames(
  matches: { name: string; kind: string }[],
  requested: string[],
  registered: ReadonlyMap<string, ReadonlySet<Kind>>,
): string[] {
  const found: string[] = [];
  for (const raw of new Set(requested)) {
    const target = resolveTarget(raw, registered);
    if (!target) continue;
    const identityRegistered = registered.get(target.name)?.has("face") || registered.get(target.name)?.has("character");
    const hit = matches.some(match => match.name === target.name && (
      target.kind ? match.kind === `${target.kind}s` :
        match.kind === "faces" || match.kind === "characters" || (!identityRegistered && match.kind === "costumes")
    ));
    if (hit) found.push(raw);
  }
  return found;
}

export async function analyzeNamedImage(buffer: Buffer, names: string[], cfg: Config): Promise<(Match & { kind: string })[]> {
  if (!names.length) return [];
  const registered = registeredKinds();
  const targets: Partial<Record<Kind, Set<string>>> = {};
  for (const raw of names) {
    const target = resolveTarget(raw, registered);
    if (!target) continue;
    for (const kind of target.kind ? [target.kind] : (registered.get(target.name) || [])) {
      (targets[kind] ||= new Set()).add(target.name);
    }
  }
  // Skip detectors and recognizers for kinds with no requested names. Competitors
  // inside an enabled kind remain available for the existing margin checks.
  // Empty OCR targets skip OCR: its independent job remains the sole OCR execution.
  const result = await analyzeImage(buffer, [], cfg, targets);
  // Keep the costume result with its kind even when the folder name also names a person.
  // The worker decides whether an untyped found name has enough identity evidence.
  return selectNamedMatches(result, names, registered);
}
