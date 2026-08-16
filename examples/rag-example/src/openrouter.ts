import type { Message, EmbeddingResponse, ChatCompletionResponse, RerankResponse } from './types';

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1';

// Free-tier models on OpenRouter — the demo runs at zero cost.
export const DEFAULT_EMBEDDING_MODEL = 'nvidia/llama-nemotron-embed-vl-1b-v2:free';
const DEFAULT_CHAT_MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free';
const DEFAULT_RERANK_MODEL = 'nvidia/llama-nemotron-rerank-vl-1b-v2:free';

const OPENROUTER_HEADERS = (apiKey: string): Record<string, string> => ({
  'Authorization': `Bearer ${apiKey}`,
  'Content-Type': 'application/json',
  'HTTP-Referer': window.location.origin,
  'X-Title': 'LatticeDB RAG Example'
});

/**
 * Parse an OpenRouter response, failing fast with a readable message.
 *
 * OpenRouter can report upstream failures as **HTTP 200 with an `error` body**
 * (e.g. `{"error":{"message":"Upstream error from Nvidia: ResourceExhausted…"}}`),
 * which is common on the free tier when a provider is saturated. Without this
 * check the caller reads a missing field and throws an opaque TypeError, so
 * every request funnels through here.
 */
async function parseResponse<T>(response: Response, what: string): Promise<T> {
  const raw = await response.text();

  if (!response.ok) {
    throw new TransientAwareError(
      `${what} failed: ${response.status} - ${raw}`,
      response.status === 429 || response.status >= 500
    );
  }

  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new TransientAwareError(`${what} returned invalid JSON: ${raw.slice(0, 200)}`, false);
  }

  const maybeError = (data as { error?: { message?: string; code?: number } }).error;
  if (maybeError) {
    const code = maybeError.code ?? 0;
    const message = maybeError.message ?? 'unknown error';
    throw new TransientAwareError(
      `${what} failed${code ? ` (${code})` : ''}: ${message}`,
      code === 429 || code >= 500 || /ResourceExhausted|rate.?limit|overloaded/i.test(message)
    );
  }

  return data as T;
}

/** Error that knows whether retrying could plausibly help. */
class TransientAwareError extends Error {
  constructor(message: string, readonly transient: boolean) {
    super(message);
    this.name = 'OpenRouterError';
  }
}

// The free tier shares provider capacity, so requests intermittently come back
// as "ResourceExhausted". A couple of short retries turn most of those into a
// successful call instead of a failed answer.
const MAX_ATTEMPTS = 3;
const RETRY_BASE_MS = 700;

async function withRetry<T>(operation: () => Promise<T>): Promise<T> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      const transient = error instanceof TransientAwareError && error.transient;
      if (!transient || attempt === MAX_ATTEMPTS) break;
      // Exponential backoff: 700ms, 1400ms
      await new Promise(resolve => setTimeout(resolve, RETRY_BASE_MS * 2 ** (attempt - 1)));
    }
  }

  throw lastError;
}

// Embed a batch of texts in a single request. Results are returned in the same
// order as `texts` (the API may return them out of order, so we sort by index).
export async function getEmbeddings(
  texts: string[],
  apiKey: string,
  model: string = DEFAULT_EMBEDDING_MODEL
): Promise<number[][]> {
  if (texts.length === 0) return [];

  const data = await withRetry(async () => {
    const response = await fetch(`${OPENROUTER_API_URL}/embeddings`, {
      method: 'POST',
      headers: OPENROUTER_HEADERS(apiKey),
      body: JSON.stringify({
        model,
        input: texts
      })
    });
    return parseResponse<EmbeddingResponse>(response, 'Embedding request');
  });

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

// Prepend the RAG system prompt (with retrieved context) to the conversation.
function withSystemPrompt(messages: Message[], context: string): Message[] {
  return [
    {
      role: 'system',
      content: `You are a helpful assistant. Use the following context to answer questions accurately and concisely. If the context doesn't contain relevant information, say so.

Context:
${context}`
    },
    ...messages
  ];
}

export async function chat(
  messages: Message[],
  context: string,
  apiKey: string,
  model: string = DEFAULT_CHAT_MODEL
): Promise<string> {
  const allMessages = withSystemPrompt(messages, context);

  const data = await withRetry(async () => {
    const response = await fetch(`${OPENROUTER_API_URL}/chat/completions`, {
      method: 'POST',
      headers: OPENROUTER_HEADERS(apiKey),
      body: JSON.stringify({
        model,
        messages: allMessages
      })
    });
    return parseResponse<ChatCompletionResponse>(response, 'Chat request');
  });
  const content = data.choices?.[0]?.message?.content;
  if (content === undefined) {
    throw new Error('Chat request returned no message content');
  }
  return content;
}

/// A single streamed increment: reasoning ("thinking") and/or answer tokens.
export interface ChatStreamDelta {
  reasoning?: string;
  content?: string;
}

/**
 * Stream a chat completion, invoking `onDelta` as reasoning and answer tokens
 * arrive. Resolves with the full answer text (reasoning excluded).
 *
 * OpenRouter emits Server-Sent Events: `data:` lines carrying a JSON chunk,
 * `:`-prefixed keepalive comments, and a final `data: [DONE]`. Reasoning
 * models put thinking on `delta.reasoning` while `delta.content` is empty, then
 * switch to `delta.content` for the answer. An upstream failure can also arrive
 * as an `{ "error": ... }` chunk mid-stream.
 */
export async function chatStream(
  messages: Message[],
  context: string,
  apiKey: string,
  onDelta: (delta: ChatStreamDelta) => void,
  model: string = DEFAULT_CHAT_MODEL
): Promise<string> {
  const allMessages = withSystemPrompt(messages, context);
  const body = JSON.stringify({ model, messages: allMessages, stream: true });

  let content = '';
  let emitted = false;

  const runOnce = async (): Promise<void> => {
    const response = await fetch(`${OPENROUTER_API_URL}/chat/completions`, {
      method: 'POST',
      headers: OPENROUTER_HEADERS(apiKey),
      body
    });

    if (!response.ok) {
      const text = await response.text();
      throw new TransientAwareError(
        `Chat request failed: ${response.status} - ${text}`,
        response.status === 429 || response.status >= 500
      );
    }
    if (!response.body) {
      throw new TransientAwareError('Chat request returned no response stream', true);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are newline-delimited; process every complete line and keep
      // any partial remainder in the buffer.
      let newlineIdx: number;
      while ((newlineIdx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);

        if (line === '' || line.startsWith(':')) continue; // blank / keepalive
        if (!line.startsWith('data:')) continue;

        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return;

        let chunk: {
          error?: { message?: string; code?: number };
          choices?: Array<{ delta?: { reasoning?: unknown; content?: unknown } }>;
        };
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue; // ignore any non-JSON line
        }

        if (chunk.error) {
          const message = chunk.error.message ?? 'unknown error';
          throw new TransientAwareError(
            `Chat request failed: ${message}`,
            (chunk.error.code ?? 0) >= 500 ||
              /ResourceExhausted|rate.?limit|overloaded/i.test(message)
          );
        }

        const delta = chunk.choices?.[0]?.delta;
        if (!delta) continue;

        const reasoning = typeof delta.reasoning === 'string' && delta.reasoning
          ? delta.reasoning
          : undefined;
        const answer = typeof delta.content === 'string' && delta.content
          ? delta.content
          : undefined;

        if (reasoning || answer) {
          emitted = true;
          if (answer) content += answer;
          onDelta({ reasoning, content: answer });
        }
      }
    }
  };

  // Retry only while nothing has been emitted yet: once tokens have been
  // rendered, restarting the stream would duplicate them.
  for (let attempt = 1; ; attempt++) {
    try {
      await runOnce();
      return content;
    } catch (error) {
      const transient = error instanceof TransientAwareError && error.transient;
      if (!transient || emitted || attempt >= MAX_ATTEMPTS) throw error;
      await new Promise(resolve => setTimeout(resolve, RETRY_BASE_MS * 2 ** (attempt - 1)));
    }
  }
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
  const data = await withRetry(async () => {
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
    return parseResponse<RerankResponse>(response, 'Rerank request');
  });

  return data.results
    .slice()
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .map(r => ({ index: r.index, score: r.relevance_score }));
}
