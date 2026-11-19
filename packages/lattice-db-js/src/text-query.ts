/**
 * Text queries for fetch hosts.
 *
 * Qdrant's inference form lets a client send text instead of a vector:
 * `POST /collections/{c}/points/query` with
 * `{ "query": { "text": "...", "model": "..." } }` (or nested under
 * `"nearest"`). The router only understands vectors, so a host either embeds
 * the text with a configured {@link Embedder} and forwards a vector query, or
 * answers 501 — never a silent wrong answer.
 */

/** Turn texts into vectors with the named model; one vector per text, in order. */
export type Embedder = (model: string, texts: string[]) => Promise<Float32Array[]>;

/** What a host should do with an incoming request body. */
export type TextQueryOutcome =
  | { kind: 'pass' }
  | { kind: 'rewrite'; body: string }
  | { kind: 'reject'; status: 400 | 501; message: string };

interface TextQuery {
  text: string;
  model: string | undefined;
}

const QUERY_PATH = /\/collections\/[^/]+\/points\/query$/;

/** Whether this request can carry a text query (only then is the body read). */
export function mayCarryTextQuery(method: string, pathname: string): boolean {
  return method.toUpperCase() === 'POST' && QUERY_PATH.test(pathname);
}

/**
 * Decide how to handle a query body. Non-JSON or vector queries pass through
 * untouched for the router to judge.
 */
export async function resolveTextQuery(
  bodyText: string,
  embed: Embedder | undefined
): Promise<TextQueryOutcome> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    return { kind: 'pass' };
  }
  if (!isRecord(parsed)) return { kind: 'pass' };
  const textQuery = findTextQuery(parsed.query);
  if (textQuery === null) return { kind: 'pass' };
  if (embed === undefined) {
    return {
      kind: 'reject',
      status: 501,
      message: 'Text queries need an embedder; configure the embed option or send a vector',
    };
  }
  if (textQuery.model === undefined) {
    return { kind: 'reject', status: 400, message: 'A text query must name its model' };
  }
  const vectors = await embed(textQuery.model, [textQuery.text]);
  if (vectors.length !== 1) {
    throw new Error(`Embedder returned ${vectors.length} vectors for 1 text`);
  }
  const rewritten = { ...parsed, query: Array.from(vectors[0]) };
  return { kind: 'rewrite', body: JSON.stringify(rewritten) };
}

function findTextQuery(query: unknown): TextQuery | null {
  if (!isRecord(query)) return null;
  const candidate = isRecord(query.nearest) ? query.nearest : query;
  if (typeof candidate.text !== 'string') return null;
  const model = typeof candidate.model === 'string' ? candidate.model : undefined;
  return { text: candidate.text, model };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
