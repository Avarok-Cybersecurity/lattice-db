/**
 * LatticeDB - Hybrid Graph/Vector Database for the Browser
 *
 * @packageDocumentation
 * @module lattice-db
 *
 * @example
 * ```typescript
 * import { LatticeDB } from 'lattice-db';
 *
 * const db = await LatticeDB.init();
 *
 * // Create collection
 * db.createCollection('products', {
 *   vectors: { size: 384, distance: 'Cosine' }
 * });
 *
 * // Insert data
 * db.upsert('products', [
 *   { id: 1, vector: embedding, payload: { name: 'Widget' } }
 * ]);
 *
 * // Search
 * const results = db.search('products', queryVector, 10);
 * ```
 */

// Main database class
export { LatticeDB, LatticeDB as default } from './lattice';

// All type definitions
export type {
  // Configuration types
  DistanceMetric,
  VectorParams,
  HnswConfig,
  CollectionConfig,
  CollectionInfo,

  // Point types
  Payload,
  Point,
  PointRecord,

  // Search types
  SearchResult,
  SearchOptions,
  ScrollOptions,
  ScrollResult,

  // Operation results
  UpsertResult,
  UpdateResult,

  // Graph types
  TraversalNode,
  TraversalEdge,
  TraversalResult,
  CypherResult,

  // API wrapper
  ApiResponse,
} from './types';
