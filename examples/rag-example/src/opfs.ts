/**
 * Persistent snapshot storage backed by the Origin Private File System (OPFS).
 *
 * All OPFS access lives here so the rest of the app stays free of I/O.
 *
 * Layout — one directory per snapshot under `lattice-db/`:
 *
 *   lattice-db/<id>/meta.json   small header: label, counts, timestamps
 *   lattice-db/<id>/docs.json   document text + metadata (no vectors)
 *   lattice-db/<id>/vecs.bin    contiguous Float32 vectors, row-major
 *
 * Vectors are written as raw Float32 rather than JSON: a 492-document corpus at
 * 2048 dimensions is ~4 MB binary versus ~10 MB of JSON text, and it restores
 * without parsing a million numbers.
 */

import type { DocumentSource } from './types';

/// Bumped whenever the on-disk layout changes. Snapshots written by a different
/// version are surfaced but refused rather than misread.
export const SNAPSHOT_SCHEMA_VERSION = 1;

const ROOT_DIR = 'lattice-db';
const META_FILE = 'meta.json';
const DOCS_FILE = 'docs.json';
const VECS_FILE = 'vecs.bin';

export interface SnapshotMeta {
  id: string;
  label: string;
  schemaVersion: number;
  /** Incremented on every auto-save, so the UI can show how much has changed. */
  revision: number;
  createdAt: number;
  updatedAt: number;
  documentCount: number;
  vectorDimension: number;
  embeddingModel: string;
  /** Document counts per source, e.g. `{ docs: 492, file: 3 }`. */
  sources: Record<string, number>;
  /** Total bytes on disk; filled in by `listSnapshots`. */
  bytes: number;
}

export interface PersistedDocument {
  id: number;
  text: string;
  title: string;
  source: DocumentSource;
  addedAt: number;
  metadata?: Record<string, unknown>;
}

export interface Snapshot {
  meta: SnapshotMeta;
  documents: PersistedDocument[];
  /** Row-major vectors: document `i` occupies `[i*dim, (i+1)*dim)`. */
  vectors: Float32Array;
}

/** True when the browser exposes OPFS. */
export function isSupported(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory;
}

async function rootDir(create: boolean): Promise<FileSystemDirectoryHandle | null> {
  if (!isSupported()) return null;
  const root = await navigator.storage.getDirectory();
  try {
    return await root.getDirectoryHandle(ROOT_DIR, { create });
  } catch {
    return null;
  }
}

async function readFile(dir: FileSystemDirectoryHandle, name: string): Promise<File | null> {
  try {
    const handle = await dir.getFileHandle(name);
    return await handle.getFile();
  } catch {
    return null;
  }
}

async function writeFile(
  dir: FileSystemDirectoryHandle,
  name: string,
  data: string | ArrayBuffer | ArrayBufferView
): Promise<void> {
  const handle = await dir.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  try {
    await writable.write(data as FileSystemWriteChunkType);
    await writable.close();
  } catch (error) {
    await writable.abort().catch(() => undefined);
    throw error;
  }
}

/**
 * List every stored snapshot, newest first.
 *
 * Only the small `meta.json` of each snapshot is parsed, so this stays fast
 * even when the payloads are large.
 */
export async function listSnapshots(): Promise<SnapshotMeta[]> {
  const dir = await rootDir(false);
  if (!dir) return [];

  const metas: SnapshotMeta[] = [];
  for await (const [id, handle] of dir as unknown as AsyncIterable<
    [string, FileSystemHandle]
  >) {
    if (handle.kind !== 'directory') continue;
    const snapshotDir = handle as FileSystemDirectoryHandle;

    const metaFile = await readFile(snapshotDir, META_FILE);
    if (!metaFile) continue;

    try {
      const meta = JSON.parse(await metaFile.text()) as SnapshotMeta;
      const docs = await readFile(snapshotDir, DOCS_FILE);
      const vecs = await readFile(snapshotDir, VECS_FILE);
      metas.push({
        ...meta,
        id,
        bytes: metaFile.size + (docs?.size ?? 0) + (vecs?.size ?? 0)
      });
    } catch {
      // Unreadable snapshot: skip it rather than breaking the listing.
    }
  }

  return metas.sort((a, b) => b.updatedAt - a.updatedAt);
}

/** Write (or overwrite) a snapshot. */
export async function saveSnapshot(
  meta: Omit<SnapshotMeta, 'bytes' | 'schemaVersion'>,
  documents: PersistedDocument[],
  vectors: Float32Array
): Promise<void> {
  const dir = await rootDir(true);
  if (!dir) throw new Error('OPFS is not available in this browser');

  const snapshotDir = await dir.getDirectoryHandle(meta.id, { create: true });
  const header: SnapshotMeta = {
    ...meta,
    schemaVersion: SNAPSHOT_SCHEMA_VERSION,
    bytes: 0
  };

  // Payload first: if writing is interrupted, meta.json is not left claiming
  // data that never landed.
  await writeFile(snapshotDir, DOCS_FILE, JSON.stringify(documents));
  await writeFile(snapshotDir, VECS_FILE, vectors.buffer as ArrayBuffer);
  await writeFile(snapshotDir, META_FILE, JSON.stringify(header));
}

/** Read a snapshot back. Returns null if it is missing or unreadable. */
export async function readSnapshot(id: string): Promise<Snapshot | null> {
  const dir = await rootDir(false);
  if (!dir) return null;

  let snapshotDir: FileSystemDirectoryHandle;
  try {
    snapshotDir = await dir.getDirectoryHandle(id);
  } catch {
    return null;
  }

  const metaFile = await readFile(snapshotDir, META_FILE);
  const docsFile = await readFile(snapshotDir, DOCS_FILE);
  const vecsFile = await readFile(snapshotDir, VECS_FILE);
  if (!metaFile || !docsFile || !vecsFile) return null;

  const meta = JSON.parse(await metaFile.text()) as SnapshotMeta;
  if (meta.schemaVersion !== SNAPSHOT_SCHEMA_VERSION) {
    throw new Error(
      `Snapshot "${meta.label}" uses format v${meta.schemaVersion}; this build reads v${SNAPSHOT_SCHEMA_VERSION}`
    );
  }

  const documents = JSON.parse(await docsFile.text()) as PersistedDocument[];
  const vectors = new Float32Array(await vecsFile.arrayBuffer());

  const expected = documents.length * meta.vectorDimension;
  if (vectors.length !== expected) {
    throw new Error(
      `Snapshot "${meta.label}" is corrupt: expected ${expected} vector values, found ${vectors.length}`
    );
  }

  return { meta: { ...meta, id, bytes: metaFile.size + docsFile.size + vecsFile.size }, documents, vectors };
}

export async function deleteSnapshot(id: string): Promise<void> {
  const dir = await rootDir(false);
  if (!dir) return;
  await dir.removeEntry(id, { recursive: true }).catch(() => undefined);
}

export async function clearAll(): Promise<void> {
  if (!isSupported()) return;
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(ROOT_DIR, { recursive: true }).catch(() => undefined);
}

/** Total bytes used by this app's snapshots. */
export async function totalBytes(): Promise<number> {
  const metas = await listSnapshots();
  return metas.reduce((sum, m) => sum + m.bytes, 0);
}

/** Human-readable byte size, e.g. `4.2 MB`. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}
