import { LatticeDB } from 'lattice-db';
import { getEmbedding, getEmbeddings, chat, chatStream, rerank } from './openrouter';
import type { ChatStreamDelta } from './openrouter';
import type { Message, Document, SearchResult, RAGConfig, ManagedDocument, DocumentSource } from './types';

const COLLECTION_NAME = 'documents';
// How many vector-search candidates to rerank per requested result. The
// reranker is a cross-encoder that is far more precise than cosine similarity
// but too costly to run over the whole corpus, so we over-fetch by this factor
// then let it pick the final topK.
const RERANK_CANDIDATE_MULTIPLIER = 4;
// Texts embedded per request when bulk-loading. Batching keeps the free-tier
// request count low (hundreds of chunks -> a handful of calls).
const EMBED_BATCH_SIZE = 32;
// Points fetched per scroll page when exporting a snapshot.
const SCROLL_PAGE_SIZE = 256;

export class RAGEngine {
  private db: LatticeDB | null = null;
  private config: RAGConfig;
  private documents: Map<number, ManagedDocument> = new Map();
  // Derived from the model's first embedding — the model is the single
  // source of truth for vector size (e.g. 2048 for the NVIDIA Nemotron embed
  // model), so we never hardcode a dimension that a model swap could break.
  private embeddingDimension: number | null = null;

  constructor(config: RAGConfig) {
    this.config = {
      topK: 3,
      ...config
    };
  }

  async init(): Promise<void> {
    this.db = await LatticeDB.init();
  }

  // The WASM binding returns collection info as `{ collections: [{ name }] }`
  // at runtime (its .d.ts claims `string[]`), so normalize both shapes.
  private collectionExists(): boolean {
    if (!this.db) return false;
    const raw = this.db.listCollections() as unknown;
    const names: string[] = Array.isArray(raw)
      ? raw.map(c => (typeof c === 'string' ? c : c?.name))
      : ((raw as { collections?: Array<{ name: string }> })?.collections ?? []).map(c => c.name);
    return names.includes(COLLECTION_NAME);
  }

  private ensureCollection(size: number): void {
    if (!this.db) {
      throw new Error('RAGEngine not initialized. Call init() first.');
    }
    if (this.collectionExists()) {
      return;
    }
    this.embeddingDimension = size;
    this.db.createCollection(COLLECTION_NAME, {
      vectors: {
        size,
        distance: 'Cosine'
      }
    });
  }

  async addDocument(
    doc: Document,
    source: DocumentSource = 'manual',
    title?: string
  ): Promise<void> {
    if (!this.db) {
      throw new Error('RAGEngine not initialized. Call init() first.');
    }

    const embedding = await getEmbedding(
      doc.text,
      this.config.apiKey,
      this.config.embeddingModel
    );

    this.ensureCollection(embedding.length);

    this.db.upsert(COLLECTION_NAME, [{
      id: doc.id,
      vector: embedding,
      payload: {
        text: doc.text,
        ...doc.metadata
      }
    }]);

    this.recordDocument(doc, source, title);
  }

  // Bulk-add documents, embedding in batches to minimize API round-trips.
  // `onProgress(done, total)` fires after each batch is persisted.
  async addDocuments(
    docs: Document[],
    source: DocumentSource = 'manual',
    titleFn?: (doc: Document) => string,
    onProgress?: (done: number, total: number) => void
  ): Promise<void> {
    if (!this.db) {
      throw new Error('RAGEngine not initialized. Call init() first.');
    }

    for (let i = 0; i < docs.length; i += EMBED_BATCH_SIZE) {
      const batch = docs.slice(i, i + EMBED_BATCH_SIZE);
      const embeddings = await getEmbeddings(
        batch.map(d => d.text),
        this.config.apiKey,
        this.config.embeddingModel
      );

      this.ensureCollection(embeddings[0].length);

      this.db.upsert(COLLECTION_NAME, batch.map((doc, j) => ({
        id: doc.id,
        vector: embeddings[j],
        payload: {
          text: doc.text,
          ...doc.metadata
        }
      })));

      for (const doc of batch) {
        this.recordDocument(doc, source, titleFn ? titleFn(doc) : undefined);
      }

      onProgress?.(Math.min(i + batch.length, docs.length), docs.length);
    }
  }

  private recordDocument(doc: Document, source: DocumentSource, title?: string): void {
    this.documents.set(doc.id, {
      ...doc,
      source,
      title: title ?? doc.text.slice(0, 50) + (doc.text.length > 50 ? '...' : ''),
      addedAt: Date.now()
    });
  }

  removeDocument(id: number): boolean {
    if (!this.db) return false;

    try {
      this.db.deletePoints(COLLECTION_NAME, [id]);
      this.documents.delete(id);
      return true;
    } catch {
      return false;
    }
  }

  getDocuments(): ManagedDocument[] {
    return Array.from(this.documents.values()).sort((a, b) => b.addedAt - a.addedAt);
  }

  /** Vector dimension in use, or null before the first document is added. */
  getVectorDimension(): number | null {
    return this.embeddingDimension;
  }

  /**
   * Read every document back out of the database together with its vector,
   * ready to be persisted. Vectors are returned row-major so they can be
   * written as one contiguous binary blob.
   */
  exportSnapshot(): { documents: ManagedDocument[]; vectors: Float32Array } | null {
    if (!this.db || !this.collectionExists() || this.embeddingDimension === null) {
      return null;
    }

    const dim = this.embeddingDimension;
    const documents: ManagedDocument[] = [];
    const chunks: number[][] = [];

    // Pull points in pages so a large corpus doesn't need one huge request.
    let offset: number | undefined;
    for (;;) {
      const page = this.db.scroll(COLLECTION_NAME, {
        limit: SCROLL_PAGE_SIZE,
        offset,
        with_payload: false,
        with_vector: true
      });

      for (const point of page.points) {
        const id = Number(point.id);
        const managed = this.documents.get(id);
        const vector = point.vector;
        // Only persist points we still have metadata for and a usable vector.
        if (!managed || !vector || vector.length !== dim) continue;
        documents.push(managed);
        chunks.push(Array.from(vector));
      }

      const next = page.next_page_offset;
      if (next === null || next === undefined) break;
      offset = Number(next);
    }

    if (documents.length === 0) return null;

    const vectors = new Float32Array(documents.length * dim);
    chunks.forEach((vec, i) => vectors.set(vec, i * dim));

    return { documents, vectors };
  }

  /**
   * Replace the current contents with a previously exported snapshot.
   *
   * No embedding calls are made — the stored vectors are inserted directly,
   * which is what makes restoring free and instant.
   */
  restoreSnapshot(
    documents: ManagedDocument[],
    vectors: Float32Array,
    dimension: number
  ): void {
    if (!this.db) {
      throw new Error('RAGEngine not initialized. Call init() first.');
    }
    if (vectors.length !== documents.length * dimension) {
      throw new Error('Snapshot vectors do not match the document count');
    }

    this.clearDocuments();
    this.ensureCollection(dimension);

    this.db.upsert(
      COLLECTION_NAME,
      documents.map((doc, i) => ({
        id: doc.id,
        vector: Array.from(vectors.subarray(i * dimension, (i + 1) * dimension)),
        payload: {
          text: doc.text,
          ...doc.metadata
        }
      }))
    );

    for (const doc of documents) {
      this.documents.set(doc.id, doc);
    }
  }

  async search(queryText: string, topK?: number): Promise<SearchResult[]> {
    if (!this.db) {
      throw new Error('RAGEngine not initialized. Call init() first.');
    }

    const k = topK ?? this.config.topK ?? 3;

    const queryVector = await getEmbedding(
      queryText,
      this.config.apiKey,
      this.config.embeddingModel
    );

    // Stage 1 — cheap recall: over-fetch candidates by cosine similarity.
    const candidateCount = Math.max(
      k,
      this.config.rerankCandidates ?? k * RERANK_CANDIDATE_MULTIPLIER
    );
    const candidates: SearchResult[] = this.db.search(
      COLLECTION_NAME,
      queryVector,
      candidateCount,
      { with_payload: true }
    ).map(r => ({
      id: Number(r.id),
      text: (r.payload?.text as string) ?? '',
      score: r.score,
      metadata: r.payload
    }));

    // Nothing to reorder — skip the rerank round-trip.
    if (candidates.length <= 1) {
      return candidates.slice(0, k);
    }

    // Stage 2 — precise reordering: a cross-encoder reranker scores each
    // candidate against the query and we keep the top k.
    const ranked = await rerank(
      queryText,
      candidates.map(c => c.text),
      this.config.apiKey,
      k,
      this.config.rerankModel
    );

    return ranked
      .slice(0, k)
      .map(({ index, score }) => ({ ...candidates[index], score }));
  }

  private buildContext(sources: SearchResult[]): string {
    return sources.length > 0
      ? sources.map((s, i) => `[${i + 1}] ${s.text}`).join('\n\n')
      : 'No relevant documents found in the knowledge base.';
  }

  async query(
    question: string,
    history: Message[] = [],
    topK?: number
  ): Promise<{ answer: string; sources: SearchResult[] }> {
    const sources = await this.search(question, topK);

    const answer = await chat(
      [...history, { role: 'user', content: question }],
      this.buildContext(sources),
      this.config.apiKey,
      this.config.chatModel
    );

    return { answer, sources };
  }

  /**
   * Streaming variant of {@link query}.
   *
   * `onSources` fires once, as soon as retrieval + reranking finish (before the
   * model is called), so the UI can show what it's grounding on immediately.
   * `onDelta` then fires for each reasoning/answer increment. Resolves with the
   * final answer and the sources used.
   */
  async queryStream(
    question: string,
    history: Message[],
    onDelta: (delta: ChatStreamDelta) => void,
    onSources?: (sources: SearchResult[]) => void,
    topK?: number
  ): Promise<{ answer: string; sources: SearchResult[] }> {
    const sources = await this.search(question, topK);
    onSources?.(sources);

    const answer = await chatStream(
      [...history, { role: 'user', content: question }],
      this.buildContext(sources),
      this.config.apiKey,
      onDelta,
      this.config.chatModel
    );

    return { answer, sources };
  }

  getDocumentCount(): number {
    return this.documents.size;
  }

  clearDocuments(): void {
    if (this.db && this.collectionExists()) {
      this.db.deleteCollection(COLLECTION_NAME);
      // Recreate eagerly only if we already know the dimension; otherwise the
      // collection is re-created lazily on the next addDocument().
      if (this.embeddingDimension !== null) {
        this.db.createCollection(COLLECTION_NAME, {
          vectors: {
            size: this.embeddingDimension,
            distance: 'Cosine'
          }
        });
      }
    }
    this.documents.clear();
  }
}
