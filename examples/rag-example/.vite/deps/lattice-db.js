import "./chunk-V6TY7KAL.js";

// node_modules/lattice-db/dist/lattice-db.esm.js
var wasmModule = null;
var initialized = false;
var LatticeDB = class _LatticeDB {
  constructor(db) {
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
  static async init(wasmPath) {
    if (!initialized) {
      wasmModule = await import("./lattice_server-Q6DPRQB4.js");
      await wasmModule.default(wasmPath);
      initialized = true;
    }
    if (!wasmModule) {
      throw new Error("WASM module not loaded");
    }
    const db = new wasmModule.LatticeDB();
    return new _LatticeDB(db);
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
  createCollection(name, config) {
    try {
      const result = this.db.createCollection(name, config);
      if (!result || result.status !== "ok" || result.result !== true) {
        const errorMsg = (result == null ? void 0 : result.error) || `status=${result == null ? void 0 : result.status}, result=${result == null ? void 0 : result.result}`;
        throw new Error(`Failed to create collection: ${name} - ${errorMsg}`);
      }
    } catch (e) {
      if (e instanceof Error) throw e;
      throw new Error(`Failed to create collection: ${name} - ${String(e)}`);
    }
  }
  /**
   * List all collections
   *
   * @returns Array of collection names
   */
  listCollections() {
    const result = this.db.listCollections();
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to list collections: ${result.error || "unknown error"}`);
    }
    return result.result;
  }
  /**
   * Get collection info
   *
   * @param name - Collection name
   * @returns Collection info including point count and configuration
   */
  getCollection(name) {
    const result = this.db.getCollection(name);
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to get collection '${name}': ${result.error || "unknown error"}`);
    }
    return result.result;
  }
  /**
   * Delete a collection
   *
   * @param name - Collection name to delete
   * @returns true if deleted, false if not found
   */
  deleteCollection(name) {
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
  upsert(collection, points) {
    const result = this.db.upsert(collection, points);
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to upsert into '${collection}': ${result.error || "unknown error"}`);
    }
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
  getPoints(collection, ids, options) {
    const idsArray = new BigUint64Array(ids.map((id) => BigInt(id)));
    const result = this.db.getPoints(
      collection,
      idsArray,
      (options == null ? void 0 : options.withPayload) ?? true,
      (options == null ? void 0 : options.withVector) ?? false
    );
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to get points from '${collection}': ${result.error || "unknown error"}`);
    }
    return result.result;
  }
  /**
   * Delete points by IDs
   *
   * @param collection - Collection name
   * @param ids - Array of point IDs to delete
   * @returns Number of points deleted
   */
  deletePoints(collection, ids) {
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
  search(collection, vector, limit, options) {
    const vectorArray = new Float32Array(vector);
    const result = this.db.search(collection, vectorArray, limit, options);
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to search '${collection}': ${result.error || "unknown error"}`);
    }
    return result.result;
  }
  /**
   * Scroll through all points with pagination
   *
   * @param collection - Collection name
   * @param options - Scroll options (limit, offset, etc.)
   * @returns Scroll result with points and next offset
   */
  scroll(collection, options) {
    const result = this.db.scroll(collection, options);
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to scroll '${collection}': ${result.error || "unknown error"}`);
    }
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
  addEdge(collection, fromId, toId, relation, weight) {
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
  traverse(collection, startId, maxDepth, relations) {
    const result = this.db.traverse(collection, BigInt(startId), maxDepth, relations);
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to traverse '${collection}': ${result.error || "unknown error"}`);
    }
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
  query(collection, cypher, parameters) {
    const result = this.db.query(collection, cypher, parameters);
    if (result.status !== "ok" || result.result === void 0) {
      throw new Error(`Failed to execute query on '${collection}': ${result.error || "unknown error"}`);
    }
    return result.result;
  }
};
export {
  LatticeDB,
  LatticeDB as default
};
//# sourceMappingURL=lattice-db.js.map
