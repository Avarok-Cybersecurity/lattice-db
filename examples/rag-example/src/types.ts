export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface Document {
  id: number;
  text: string;
  metadata?: Record<string, unknown>;
}

export interface SearchResult {
  id: number;
  text: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface EmbeddingResponse {
  data: Array<{
    embedding: number[];
    index: number;
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface ChatCompletionResponse {
  id: string;
  choices: Array<{
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface RerankResponse {
  results: Array<{
    index: number;
    relevance_score: number;
  }>;
  model: string;
}

export interface RAGConfig {
  apiKey: string;
  embeddingModel?: string;
  chatModel?: string;
  rerankModel?: string;
  topK?: number;
  // Number of vector-search candidates fed to the reranker before it narrows
  // to topK. Defaults to topK * a fixed multiplier (see rag.ts).
  rerankCandidates?: number;
}

export type DocumentSource = 'docs' | 'url' | 'file' | 'manual';

export interface ManagedDocument extends Document {
  source: DocumentSource;
  title: string;
  addedAt: number;
}

export interface DocChunk {
  id: string;
  section: string;
  content: string;
  metadata: Record<string, string>;
}

export interface DocsManifest {
  version: string;
  generated_at: string;
  source: string;
  chunks: DocChunk[];
}
