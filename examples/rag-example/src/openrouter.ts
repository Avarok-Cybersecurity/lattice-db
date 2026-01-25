import type { Message, EmbeddingResponse, ChatCompletionResponse } from './types';

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1';

const DEFAULT_EMBEDDING_MODEL = 'openai/text-embedding-3-small';
const DEFAULT_CHAT_MODEL = 'openai/gpt-4o-mini';

export async function getEmbedding(
  text: string,
  apiKey: string,
  model: string = DEFAULT_EMBEDDING_MODEL
): Promise<number[]> {
  const response = await fetch(`${OPENROUTER_API_URL}/embeddings`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': window.location.origin,
      'X-Title': 'LatticeDB RAG Example'
    },
    body: JSON.stringify({
      model,
      input: text
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Embedding request failed: ${response.status} - ${error}`);
  }

  const data: EmbeddingResponse = await response.json();
  return data.data[0].embedding;
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
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': window.location.origin,
      'X-Title': 'LatticeDB RAG Example'
    },
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
