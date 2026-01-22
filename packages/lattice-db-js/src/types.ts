/**
 * LatticeDB TypeScript Type Definitions
 *
 * Type-safe interfaces for the LatticeDB browser API.
 */

/**
 * Distance metric for vector similarity calculations
 */
export type DistanceMetric = 'Cosine' | 'Euclid' | 'Dot';

/**
 * Vector configuration for creating collections
 */
export interface VectorParams {
  /** Vector dimensionality */
  size: number;
  /** Distance metric for similarity calculations */
  distance: DistanceMetric;
}

/**
 * HNSW index configuration
 */
export interface HnswConfig {
  /** Max connections per node in upper layers */
  m: number;
  /** Max connections per node in layer 0 */
  m0?: number;
  /** Level multiplier */
  ml?: number;
  /** Search queue size for construction */
  ef_construct: number;
}

/**
 * Configuration for creating a collection
 */
export interface CollectionConfig {
  /** Vector configuration */
  vectors: VectorParams;
  /** Optional HNSW index configuration */
  hnsw_config?: HnswConfig;
}

/**
 * Payload data attached to points (arbitrary JSON object)
 */
export type Payload = Record<string, unknown>;

/**
 * Point to upsert into a collection
 */
export interface Point {
  /** Unique point ID */
  id: number;
  /** Vector embedding */
  vector: number[];
  /** Optional payload data */
  payload?: Payload;
}

/**
 * Point record returned from queries
 */
export interface PointRecord {
  /** Point ID */
  id: number;
  /** Vector (if requested) */
  vector?: number[];
  /** Payload (if requested) */
  payload?: Payload;
}

/**
 * Search result with score
 */
export interface SearchResult {
  /** Point ID */
  id: number;
  /** Similarity score */
  score: number;
  /** Vector (if requested) */
  vector?: number[];
  /** Payload (if requested) */
  payload?: Payload;
}

/**
 * Search options
 */
export interface SearchOptions {
  /** Include payload in results */
  with_payload?: boolean;
  /** Include vector in results */
  with_vector?: boolean;
  /** Minimum score threshold */
  score_threshold?: number;
}

/**
 * Scroll options for paginated retrieval
 */
export interface ScrollOptions {
  /** Maximum points to return */
  limit?: number;
  /** Offset for pagination (point ID to start after) */
  offset?: number;
  /** Include payload in results */
  with_payload?: boolean;
  /** Include vector in results */
  with_vector?: boolean;
}

/**
 * Scroll result with pagination info
 */
export interface ScrollResult {
  /** Retrieved points */
  points: PointRecord[];
  /** Next offset for pagination (null if no more results) */
  next_page_offset: number | null;
}

/**
 * Result from upsert operation
 */
export interface UpsertResult {
  /** Operation status */
  status: string;
  /** Operation ID */
  operation_id: number;
}

/**
 * Result from update operations
 */
export interface UpdateResult {
  /** Operation status */
  status: string;
  /** Operation ID */
  operation_id: number;
}

/**
 * Node in graph traversal result
 */
export interface TraversalNode {
  /** Node ID */
  id: number;
  /** Distance from start node */
  depth: number;
  /** Payload if available */
  payload?: Payload;
}

/**
 * Edge in graph traversal result
 */
export interface TraversalEdge {
  /** Source node ID */
  from: number;
  /** Target node ID */
  to: number;
  /** Relation type */
  relation: string;
  /** Edge weight */
  weight: number;
}

/**
 * Graph traversal result
 */
export interface TraversalResult {
  /** Visited nodes */
  nodes: TraversalNode[];
  /** Traversed edges */
  edges: TraversalEdge[];
}

/**
 * Cypher query result
 */
export interface CypherResult {
  /** Column names */
  columns: string[];
  /** Result rows (each row is an array of values) */
  rows: unknown[][];
}

/**
 * Collection info returned from getCollection
 */
export interface CollectionInfo {
  /** Collection status */
  status: string;
  /** Number of points in collection */
  points_count: number;
  /** Vector configuration */
  config: {
    vectors: VectorParams;
  };
}

/**
 * API response wrapper
 */
export interface ApiResponse<T> {
  /** Response time in seconds */
  time: number;
  /** Response status */
  status: string;
  /** Result data */
  result: T;
}
