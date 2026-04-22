import Database from "better-sqlite3";
import path from "path";
import fs from "fs";

let db: Database.Database;

export function initDB(dbDir: string): Database.Database {
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }
  const dbPath = path.join(dbDir, "embeddings.db");
  db = new Database(dbPath);

  db.exec(`
    CREATE TABLE IF NOT EXISTS character_embeddings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      image_path TEXT NOT NULL UNIQUE,
      embedding BLOB NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS face_embeddings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      image_path TEXT NOT NULL UNIQUE,
      embedding BLOB NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS costume_embeddings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      image_path TEXT NOT NULL UNIQUE,
      embedding BLOB NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS person_embeddings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      image_path TEXT NOT NULL UNIQUE,
      embedding BLOB NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  console.log(`Database initialized at ${dbPath}`);
  return db;
}

export function getDB(): Database.Database {
  if (!db) throw new Error("Database not initialized. Call initDB first.");
  return db;
}

export function isImageRegistered(table: string, imagePath: string): boolean {
  const row = getDB().prepare(`SELECT id FROM ${table} WHERE image_path = ?`).get(imagePath);
  return !!row;
}

export function insertEmbedding(table: string, name: string, imagePath: string, embedding: Float32Array): void {
  const buffer = Buffer.from(embedding.buffer);
  getDB()
    .prepare(`INSERT INTO ${table} (name, image_path, embedding) VALUES (?, ?, ?)`)
    .run(name, imagePath, buffer);
}

export interface EmbeddingRow {
  name: string;
  embedding: Buffer;
}

export function getAllEmbeddings(table: string): EmbeddingRow[] {
  return getDB().prepare(`SELECT name, embedding FROM ${table}`).all() as EmbeddingRow[];
}
