/**
 * LatticeDB - Hybrid Graph/Vector Database for the Browser
 *
 * This module provides a TypeScript wrapper around the WASM-compiled
 * LatticeDB database engine, offering a clean API for browser applications.
 *
 * @example
 * ```typescript
 * import { LatticeDB } from 'lattice-db';
 *
 * // Initialize the database
 * const db = await LatticeDB.init();
 *
 * // Create a collection for document embeddings
 * db.createCollection('docs', {
 *   vectors: { size: 384, distance: 'Cosine' }
 * });
 *
 * // Insert documents with embeddings
 * db.upsert('docs', [
 *   { id: 1, vector: embedding1, payload: { title: 'Hello World' } },
 *   { id: 2, vector: embedding2, payload: { title: 'Goodbye' } }
 * ]);
 *
 * // Search for similar documents
 * const results = db.search('docs', queryVector, 5);
 * ```
 */

import type {
  CollectionConfig,
  CollectionInfo,
  CypherResult,
  Payload,
  Point,
  PointRecord,
  ScrollOptions,
  ScrollResult,
  SearchOptions,
  SearchResult,
  TraversalResult,
  UpdateResult,
  UpsertResult,
  ApiResponse,
} from './types';

// Type for the WASM module
interface WasmLatticeDB {
  new (): WasmLatticeDB;
  createCollection(name: string, config: CollectionConfig): ApiResponse<boolean>;
  listCollections(): ApiResponse<string[]>;
  getCollection(name: string): ApiResponse<CollectionInfo>;
  deleteCollection(name: string): boolean;
  upsert(collection: string, points: Point[]): ApiResponse<UpsertResult>;
  getPoints(
    collection: string,
    ids: BigUint64Array,
    withPayload?: boolean,
    withVector?: boolean
  ): ApiResponse<PointRecord[]>;
  deletePoints(collection: string, ids: BigUint64Array): ApiResponse<UpdateResult>;
  search(
    collection: string,
    vector: Float32Array,
    limit: number,
    options?: SearchOptions
  ): ApiResponse<SearchResult[]>;
  scroll(collection: string, options?: ScrollOptions): ApiResponse<ScrollResult>;
  addEdge(
    collection: string,
    fromId: bigint,
    toId: bigint,
    relation: string,
    weight?: number
  ): void;
  traverse(
    collection: string,
    startId: bigint,
    maxDepth: number,
    relations?: string[]
  ): ApiResponse<TraversalResult>;
  query(
    collection: string,
    cypher: string,
    parameters?: Record<string, unknown>
  ): ApiResponse<CypherResult>;
}

interface WasmModule {
  default: (wasmPath?: string) => Promise<unknown>;
  LatticeDB: { new (): WasmLatticeDB };
}

// Module-level state
let wasmModule: WasmModule | null = null;
let initialized = false;

/**
 * LatticeDB - Hybrid Graph/Vector Database
 *
 * Provides vector similarity search (HNSW) combined with
 * graph queries (Cypher) in a single, browser-native database.
 */
export class LatticeDB {
  private db: WasmLatticeDB;

  private constructor(db: WasmLatticeDB) {
    this.db = db;
  }

  /**
   * Initialize LatticeDB
   *
   * Must be called before creating instances. Loads the WASM module.
   *
   * @param wasmPath - Optional path to WASM file (defaults to ./wasm/lattice_server_bg.wasm)
   * @returns A new LatticeDB instance
   *
   * @example
   * ```typescript
   * const db = await LatticeDB.init();
   * // or with custom WASM path
   * const db = await LatticeDB.init('/assets/lattice.wasm');
   * ```
   */
  static async init(wasmPath?: string): Promise<LatticeDB> {
    if (!initialized) {
      // Dynamic import for the WASM module
      wasmModule = (await import('../wasm/lattice_server.js')) as WasmModule;
      await wasmModule.default(wasmPath);
      initialized = true;
    }

    if (!wasmModule) {
      throw new Error('WASM module not loaded');
    }

    const db = new wasmModule.LatticeDB();
    return new LatticeDB(db);
  }

  // === Collections ===

  /**
   * Create a new collection
   *
   * @param name - Collection name (alphanumeric, underscores allowed)
   * @param config - Collection configuration
   *
   * @example
   * ```typescript
   * db.createCollection('products', {
   *   vectors: { size: 768, distance: 'Cosine' },
   *   hnsw_config: { m: 16, ef_construct: 200 }
   * });
   * ```
   */
  createCollection(name: string, config: CollectionConfig): void {
    const result = this.db.createCollection(name, config);
    if (!result.result) {
      throw new Error(`Failed to create collection: ${name}`);
    }
  }

  /**
   * List all collections
   *
   * @returns Array of collection names
   */
  listCollections(): string[] {
    const result = this.db.listCollections();
    return result.result;
  }

  /**
   * Get collection info
   *
   * @param name - Collection name
   * @returns Collection info including point count and configuration
   */
  getCollection(name: string): CollectionInfo {
    const result = this.db.getCollection(name);
    return result.result;
  }

  /**
   * Delete a collection
   *
   * @param name - Collection name to delete
   * @returns true if deleted, false if not found
   */
  deleteCollection(name: string): boolean {
    return this.db.deleteCollection(name);
  }

  // === Points ===

  /**
   * Upsert points into a collection
   *
   * Inserts new points or updates existing ones by ID.
   *
   * @param collection - Collection name
   * @param points - Array of points to upsert
   * @returns Upsert result
   *
   * @example
   * ```typescript
   * db.upsert('docs', [
   *   { id: 1, vector: [0.1, 0.2, ...], payload: { title: 'Doc 1' } },
   *   { id: 2, vector: [0.3, 0.4, ...], payload: { title: 'Doc 2' } }
   * ]);
   * ```
   */
  upsert(collection: string, points: Point[]): UpsertResult {
    const result = this.db.upsert(collection, points);
    return result.result;
  }

  /**
   * Get points by IDs
   *
   * @param collection - Collection name
   * @param ids - Array of point IDs to retrieve
   * @param options - Optional settings for payload/vector inclusion
   * @returns Array of point records
   */
  getPoints(
    collection: string,
    ids: number[],
    options?: { withPayload?: boolean; withVector?: boolean }
  ): PointRecord[] {
    const idsArray = new BigUint64Array(ids.map((id) => BigInt(id)));
    const result = this.db.getPoints(
      collection,
      idsArray,
      options?.withPayload ?? true,
      options?.withVector ?? false
    );
    return result.result;
  }

  /**
   * Delete points by IDs
   *
   * @param collection - Collection name
   * @param ids - Array of point IDs to delete
   * @returns Number of points deleted
   */
  deletePoints(collection: string, ids: number[]): number {
    const idsArray = new BigUint64Array(ids.map((id) => BigInt(id)));
    this.db.deletePoints(collection, idsArray);
    return ids.length;
  }

  // === Search ===

  /**
   * Search for nearest neighbors
   *
   * Uses HNSW algorithm for fast approximate nearest neighbor search.
   *
   * @param collection - Collection name
   * @param vector - Query vector
   * @param limit - Maximum results to return
   * @param options - Optional search parameters
   * @returns Array of search results with scores
   *
   * @example
   * ```typescript
   * const results = db.search('docs', queryEmbedding, 10, {
   *   with_payload: true,
   *   score_threshold: 0.7
   * });
   *
   * for (const result of results) {
   *   console.log(`ID: ${result.id}, Score: ${result.score}`);
   * }
   * ```
   */
  search(
    collection: string,
    vector: number[],
    limit: number,
    options?: SearchOptions
  ): SearchResult[] {
    const vectorArray = new Float32Array(vector);
    const result = this.db.search(collection, vectorArray, limit, options);
    return result.result;
  }

  /**
   * Scroll through all points with pagination
   *
   * @param collection - Collection name
   * @param options - Scroll options (limit, offset, etc.)
   * @returns Scroll result with points and next offset
   */
  scroll(collection: string, options?: ScrollOptions): ScrollResult {
    const result = this.db.scroll(collection, options);
    return result.result;
  }

  // === Graph ===

  /**
   * Add an edge between two points
   *
   * Creates a directed relationship in the graph layer.
   *
   * @param collection - Collection name
   * @param fromId - Source point ID
   * @param toId - Target point ID
   * @param relation - Relation type (e.g., 'KNOWS', 'SIMILAR_TO')
   * @param weight - Optional edge weight (default: 1.0)
   *
   * @example
   * ```typescript
   * // Create relationships
   * db.addEdge('people', 1, 2, 'KNOWS');
   * db.addEdge('docs', 100, 200, 'CITES', 0.8);
   * ```
   */
  addEdge(
    collection: string,
    fromId: number,
    toId: number,
    relation: string,
    weight?: number
  ): void {
    this.db.addEdge(collection, BigInt(fromId), BigInt(toId), relation, weight);
  }

  /**
   * Traverse the graph from a starting point
   *
   * Performs breadth-first traversal up to specified depth.
   *
   * @param collection - Collection name
   * @param startId - Starting point ID
   * @param maxDepth - Maximum traversal depth
   * @param relations - Optional filter for relation types
   * @returns Traversal result with nodes and edges
   *
   * @example
   * ```typescript
   * const result = db.traverse('people', 1, 3, ['KNOWS', 'WORKS_WITH']);
   * for (const node of result.nodes) {
   *   console.log(`Found: ${node.id} at depth ${node.depth}`);
   * }
   * ```
   */
  traverse(
    collection: string,
    startId: number,
    maxDepth: number,
    relations?: string[]
  ): TraversalResult {
    const result = this.db.traverse(collection, BigInt(startId), maxDepth, relations);
    return result.result;
  }

  /**
   * Execute a Cypher query
   *
   * Supports a subset of openCypher for graph pattern matching.
   *
   * @param collection - Collection name
   * @param cypher - Cypher query string
   * @param parameters - Optional query parameters
   * @returns Query result with columns and rows
   *
   * @example
   * ```typescript
   * const result = db.query('people',
   *   'MATCH (p:Person)-[:KNOWS]->(friend) WHERE p.id = $id RETURN friend.name',
   *   { id: 1 }
   * );
   *
   * for (const row of result.rows) {
   *   console.log(`Friend: ${row[0]}`);
   * }
   * ```
   */
  query(
    collection: string,
    cypher: string,
    parameters?: Record<string, unknown>
  ): CypherResult {
    const result = this.db.query(collection, cypher, parameters);
    return result.result;
  }
}

export default LatticeDB;
