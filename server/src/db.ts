/**
 * 임베딩 저장소.
 *
 * 과제(kind)와 임베딩 공간(space)을 분리해 저장한다. ArcFace / CCIP / WD-Tagger는
 * 서로 다른 기하를 가진 공간이라 코사인 값을 섞어 비교하면 무의미하다. space가 다르면
 * 애초에 조회되지 않도록 스키마 수준에서 막는다.
 */

import Database from "better-sqlite3";
import path from "path";
import fs from "fs";

/** 인식 과제. */
export type Kind = "face" | "character" | "costume";
/** 임베딩을 만든 모델. 같은 space끼리만 비교 가능하다. */
export type Space = "arcface" | "ccip" | "wdtag";

let db: Database.Database;

export function initDB(dbDir: string): Database.Database {
  fs.mkdirSync(dbDir, { recursive: true });
  const dbPath = path.join(dbDir, "embeddings.db");
  db = new Database(dbPath);

  db.exec(`
    CREATE TABLE IF NOT EXISTS embeddings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      kind TEXT NOT NULL,
      space TEXT NOT NULL,
      name TEXT NOT NULL,
      image_path TEXT NOT NULL,
      embedding BLOB NOT NULL,
      auto INTEGER NOT NULL DEFAULT 0,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      UNIQUE (kind, image_path)
    );
    CREATE INDEX IF NOT EXISTS idx_embeddings_lookup ON embeddings (kind, space);
  `);

  console.log(`✅ Database initialized - path: ${dbPath}`);
  return db;
}

export function getDB(): Database.Database {
  if (!db) throw new Error("Database not initialized. Call initDB first.");
  return db;
}

export function isImageRegistered(kind: Kind, imagePath: string): boolean {
  return !!getDB().prepare(`SELECT id FROM embeddings WHERE kind = ? AND image_path = ?`).get(kind, imagePath);
}

export function insertEmbedding(
  kind: Kind,
  space: Space,
  name: string,
  imagePath: string,
  embedding: Float32Array,
  auto = false,
): void {
  getDB()
    .prepare(
      `INSERT OR REPLACE INTO embeddings (kind, space, name, image_path, embedding, auto)
       VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .run(kind, space, name, imagePath, Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength), auto ? 1 : 0);
}

/** 한 신원에 등록된 모든 임베딩과 그 평균(centroid). */
export interface GalleryEntry {
  name: string;
  /** L2 정규화된 개별 등록 임베딩. */
  vectors: Float32Array[];
  /** vectors의 평균을 다시 L2 정규화한 것. 등록 노이즈를 상쇄한다. */
  centroid: Float32Array;
}

/**
 * better-sqlite3의 Buffer는 공용 풀의 뷰라 byteOffset이 4바이트 정렬이라는 보장이 없다.
 * ArrayBuffer.slice로 0-오프셋 복사본을 만들어야 Float32Array 생성이 안전하다.
 */
function bufferToFloat32(buf: Buffer): Float32Array {
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

export function l2normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

export function getGallery(kind: Kind, space: Space): GalleryEntry[] {
  const rows = getDB()
    .prepare(`SELECT name, embedding FROM embeddings WHERE kind = ? AND space = ?`)
    .all(kind, space) as { name: string; embedding: Buffer }[];

  const byName = new Map<string, Float32Array[]>();
  for (const row of rows) {
    const vec = l2normalize(bufferToFloat32(row.embedding));
    const list = byName.get(row.name);
    if (list) list.push(vec);
    else byName.set(row.name, [vec]);
  }

  const entries: GalleryEntry[] = [];
  for (const [name, vectors] of byName) {
    const dim = vectors[0].length;
    const sum = new Float32Array(dim);
    for (const v of vectors) {
      for (let i = 0; i < dim; i++) sum[i] += v[i];
    }
    for (let i = 0; i < dim; i++) sum[i] /= vectors.length;
    entries.push({ name, vectors, centroid: l2normalize(sum) });
  }
  return entries;
}

export function countEmbeddings(kind: Kind): number {
  const row = getDB().prepare(`SELECT COUNT(*) AS n FROM embeddings WHERE kind = ?`).get(kind) as { n: number };
  return row.n;
}
