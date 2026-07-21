# RAG Example with LatticeDB + OpenRouter

A browser-based RAG (Retrieval-Augmented Generation) application demonstrating
LatticeDB's vector search capabilities with OpenRouter for LLM inference.

## Features

- Semantic document search using vector embeddings
- Natural language Q&A with context retrieval
- Runs entirely in the browser (no backend required)
- Sub-millisecond vector search via WebAssembly

## Setup

1. Get an OpenRouter API key from https://openrouter.ai/keys

2. Install dependencies:
   ```bash
   cd examples/rag-example
   npm install
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:5173 and enter your API key

## How It Works

1. **Add documents** - Text is embedded using OpenRouter's embedding API and stored in LatticeDB
2. **Ask questions** - Your question is embedded, similar documents are retrieved via vector search
3. **Get answers** - The LLM generates a response using the retrieved context

## Tech Stack

- **LatticeDB** (WASM) - Hybrid graph/vector database
- **OpenRouter API** - Embeddings + LLM inference
- **Vite + TypeScript** - Build tooling

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   User Input    │ ──▶ │  OpenRouter API  │ ──▶ │  Embedding  │
└─────────────────┘     │  (embed text)    │     │  [2048 dim] │
                        └──────────────────┘     └──────┬──────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  LLM Response   │ ◀── │  OpenRouter API  │ ◀── │  LatticeDB  │
│  with context   │     │  (chat + RAG)    │     │  (WASM)     │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

## API Usage

```typescript
import { RAGEngine } from './rag';

// Initialize
const engine = new RAGEngine({ apiKey: 'sk-or-v1-...' });
await engine.init();

// Add documents
await engine.addDocument({
  id: 1,
  text: 'LatticeDB is a hybrid graph/vector database...'
});

// Query with RAG
const { answer, sources } = await engine.query(
  'How does LatticeDB work?'
);

console.log(answer);    // LLM-generated answer
console.log(sources);   // Retrieved documents with scores
```

## Models Used

Both defaults are **free** OpenRouter models, so the demo runs at zero cost:

- **Embeddings**: `nvidia/llama-nemotron-embed-vl-1b-v2:free` (2048 dimensions)
- **Chat**: `nvidia/nemotron-3-ultra-550b-a55b:free`

The vector dimension is derived automatically from the embedding model's
output, so swapping models does not require any code changes.

You can customize models when initializing the RAGEngine:

```typescript
const engine = new RAGEngine({
  apiKey: 'sk-or-v1-...',
  embeddingModel: 'openai/text-embedding-3-large',
  chatModel: 'anthropic/claude-3-haiku'
});
```

## License

MIT
