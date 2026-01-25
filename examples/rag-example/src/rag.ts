import { LatticeDB } from 'lattice-db';
import { getEmbedding, chat } from './openrouter';
import type { Message, Document, SearchResult, RAGConfig, ManagedDocument, DocumentSource } from './types';

const COLLECTION_NAME = 'documents';
const EMBEDDING_DIMENSION = 1536; // text-embedding-3-small

export class RAGEngine {
  private db: LatticeDB | null = null;
  private config: RAGConfig;
  private documents: Map<number, ManagedDocument> = new Map();

  constructor(config: RAGConfig) {
    this.config = {
      topK: 3,
      ...config
    };
  }

  async init(): Promise<void> {
    this.db = await LatticeDB.init();

    try {
      this.db.createCollection(COLLECTION_NAME, {
        vectors: {
          size: EMBEDDING_DIMENSION,
          distance: 'Cosine'
        }
      });
    } catch (e) {
      // Collection may already exist, try to get it
      const collections = this.db.listCollections();
      if (!collections.includes(COLLECTION_NAME)) {
        throw e;
      }
    }
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

    this.db.upsert(COLLECTION_NAME, [{
      id: doc.id,
      vector: embedding,
      payload: {
        text: doc.text,
        ...doc.metadata
      }
    }]);

    const managedDoc: ManagedDocument = {
      ...doc,
      source,
      title: title ?? doc.text.slice(0, 50) + (doc.text.length > 50 ? '...' : ''),
      addedAt: Date.now()
    };
    this.documents.set(doc.id, managedDoc);
  }

  async addDocuments(
    docs: Document[],
    source: DocumentSource = 'manual',
    titleFn?: (doc: Document) => string
  ): Promise<void> {
    for (const doc of docs) {
      const title = titleFn ? titleFn(doc) : undefined;
      await this.addDocument(doc, source, title);
    }
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

  async search(queryText: string, topK?: number): Promise<SearchResult[]> {
    if (!this.db) {
      throw new Error('RAGEngine not initialized. Call init() first.');
    }

    const queryVector = await getEmbedding(
      queryText,
      this.config.apiKey,
      this.config.embeddingModel
    );

    const results = this.db.search(
      COLLECTION_NAME,
      queryVector,
      topK ?? this.config.topK ?? 3,
      { with_payload: true }
    );

    return results.map(r => ({
      id: Number(r.id),
      text: (r.payload?.text as string) ?? '',
      score: r.score,
      metadata: r.payload
    }));
  }

  async query(
    question: string,
    history: Message[] = [],
    topK?: number
  ): Promise<{ answer: string; sources: SearchResult[] }> {
    const sources = await this.search(question, topK);

    const context = sources.length > 0
      ? sources.map((s, i) => `[${i + 1}] ${s.text}`).join('\n\n')
      : 'No relevant documents found in the knowledge base.';

    const messages: Message[] = [
      ...history,
      { role: 'user', content: question }
    ];

    const answer = await chat(
      messages,
      context,
      this.config.apiKey,
      this.config.chatModel
    );

    return { answer, sources };
  }

  getDocumentCount(): number {
    return this.documents.size;
  }

  clearDocuments(): void {
    if (this.db) {
      this.db.deleteCollection(COLLECTION_NAME);
      this.db.createCollection(COLLECTION_NAME, {
        vectors: {
          size: EMBEDDING_DIMENSION,
          distance: 'Cosine'
        }
      });
    }
    this.documents.clear();
  }
}
