import type { Message, EmbeddingResponse, ChatCompletionResponse, RerankResponse } from './types';

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1';

// Free-tier models on OpenRouter — the demo runs at zero cost.
const DEFAULT_EMBEDDING_MODEL = 'nvidia/llama-nemotron-embed-vl-1b-v2:free';
const DEFAULT_CHAT_MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free';
const DEFAULT_RERANK_MODEL = 'nvidia/llama-nemotron-rerank-vl-1b-v2:free';

const OPENROUTER_HEADERS = (apiKey: string): Record<string, string> => ({
  'Authorization': `Bearer ${apiKey}`,
  'Content-Type': 'application/json',
  'HTTP-Referer': window.location.origin,
  'X-Title': 'LatticeDB RAG Example'
});

// Embed a batch of texts in a single request. Results are returned in the same
// order as `texts` (the API may return them out of order, so we sort by index).
export async function getEmbeddings(
  texts: string[],
  apiKey: string,
  model: string = DEFAULT_EMBEDDING_MODEL
): Promise<number[][]> {
  if (texts.length === 0) return [];

  const response = await fetch(`${OPENROUTER_API_URL}/embeddings`, {
    method: 'POST',
    headers: OPENROUTER_HEADERS(apiKey),
    body: JSON.stringify({
      model,
      input: texts
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Embedding request failed: ${response.status} - ${error}`);
  }

  const data: EmbeddingResponse = await response.json();
  return data.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map(d => d.embedding);
}

export async function getEmbedding(
  text: string,
  apiKey: string,
  model: string = DEFAULT_EMBEDDING_MODEL
): Promise<number[]> {
  const [embedding] = await getEmbeddings([text], apiKey, model);
  return embedding;
}

export async function chat(
  messages: Message[],
  context: string,
  apiKey: string,
  model: string = DEFAULT_CHAT_MODEL
): Promise<string> {
  const systemMessage: Message = {
    role: 'system',
    content: `You are a helpful assistant. Use the following context to answer questions accurately and concisely. If the context doesn't contain relevant information, say so.

Context:
${context}`
  };

  const allMessages = [systemMessage, ...messages];

  const response = await fetch(`${OPENROUTER_API_URL}/chat/completions`, {
    method: 'POST',
    headers: OPENROUTER_HEADERS(apiKey),
    body: JSON.stringify({
      model,
      messages: allMessages
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Chat request failed: ${response.status} - ${error}`);
  }

  const data: ChatCompletionResponse = await response.json();
  return data.choices[0].message.content;
}

/**
 * Rerank `documents` against `query` with a cross-encoder reranker.
 * Returns document indices (into the input array) ordered most- to
 * least-relevant, each with its relevance score. Callers map indices back to
 * their own records — we do not rely on the response echoing document text.
 */
export async function rerank(
  query: string,
  documents: string[],
  apiKey: string,
  topN: number,
  model: string = DEFAULT_RERANK_MODEL
): Promise<Array<{ index: number; score: number }>> {
  const response = await fetch(`${OPENROUTER_API_URL}/rerank`, {
    method: 'POST',
    headers: OPENROUTER_HEADERS(apiKey),
    body: JSON.stringify({
      model,
      query,
      documents,
      top_n: topN,
      return_documents: false
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Rerank request failed: ${response.status} - ${error}`);
  }

  const data: RerankResponse = await response.json();
  return data.results
    .slice()
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .map(r => ({ index: r.index, score: r.relevance_score }));
}
