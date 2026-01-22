/**
 * LatticeDB TypeScript Library Tests
 *
 * These tests verify the TypeScript wrapper API using a mocked WASM module.
 * Integration tests with actual WASM run in wasm-bindgen-test (Rust).
 */

import { jest, describe, it, expect, beforeAll, beforeEach } from '@jest/globals';
import type {
  CollectionConfig,
  Point,
  SearchResult,
  PointRecord,
  ScrollResult,
  TraversalResult,
  CypherResult,
  CollectionInfo,
  UpsertResult,
  UpdateResult,
  ApiResponse,
} from '../types';

// Mock WASM module - simulates actual WASM behavior for unit tests
const mockWasmDb = {
  createCollection: jest.fn(),
  listCollections: jest.fn(),
  getCollection: jest.fn(),
  deleteCollection: jest.fn(),
  upsert: jest.fn(),
  getPoints: jest.fn(),
  deletePoints: jest.fn(),
  search: jest.fn(),
  scroll: jest.fn(),
  addEdge: jest.fn(),
  traverse: jest.fn(),
  query: jest.fn(),
};

// Mock the WASM module import
jest.mock('../../wasm/lattice_server.js', () => ({
  default: jest.fn().mockResolvedValue(undefined),
  LatticeDB: jest.fn().mockImplementation(() => mockWasmDb),
}));

// Import after mocking
import { LatticeDB } from '../lattice';

describe('LatticeDB', () => {
  let db: LatticeDB;

  beforeAll(async () => {
    db = await LatticeDB.init();
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Collections', () => {
    it('should create a collection', () => {
      const config: CollectionConfig = {
        vectors: { size: 128, distance: 'Cosine' },
      };

      mockWasmDb.createCollection.mockReturnValue({
        result: true,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<boolean>);

      expect(() => db.createCollection('test_collection', config)).not.toThrow();
      expect(mockWasmDb.createCollection).toHaveBeenCalledWith('test_collection', config);
    });

    it('should list collections', () => {
      mockWasmDb.listCollections.mockReturnValue({
        result: ['collection1', 'collection2'],
        status: 'ok',
        time: 0.001,
      } as ApiResponse<string[]>);

      const collections = db.listCollections();

      expect(collections).toEqual(['collection1', 'collection2']);
      expect(mockWasmDb.listCollections).toHaveBeenCalled();
    });

    it('should get collection info', () => {
      const info: CollectionInfo = {
        status: 'green',
        points_count: 100,
        config: { vectors: { size: 128, distance: 'Cosine' } },
      };

      mockWasmDb.getCollection.mockReturnValue({
        result: info,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<CollectionInfo>);

      const result = db.getCollection('test');

      expect(result).toEqual(info);
      expect(mockWasmDb.getCollection).toHaveBeenCalledWith('test');
    });

    it('should delete a collection', () => {
      mockWasmDb.deleteCollection.mockReturnValue(true);

      const deleted = db.deleteCollection('to_delete');

      expect(deleted).toBe(true);
      expect(mockWasmDb.deleteCollection).toHaveBeenCalledWith('to_delete');
    });

    it('should return false when deleting non-existent collection', () => {
      mockWasmDb.deleteCollection.mockReturnValue(false);

      const deleted = db.deleteCollection('nonexistent');

      expect(deleted).toBe(false);
    });
  });

  describe('Points', () => {
    it('should upsert points', () => {
      const points: Point[] = [
        { id: 1, vector: [0.1, 0.2, 0.3, 0.4], payload: { name: 'Alice' } },
        { id: 2, vector: [0.5, 0.6, 0.7, 0.8], payload: { name: 'Bob' } },
      ];

      const upsertResult: UpsertResult = {
        status: 'completed',
        operation_id: 0,
      };

      mockWasmDb.upsert.mockReturnValue({
        result: upsertResult,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<UpsertResult>);

      const result = db.upsert('test', points);

      expect(result.status).toBe('completed');
      expect(mockWasmDb.upsert).toHaveBeenCalledWith('test', points);
    });

    it('should get points by ids', () => {
      const records: PointRecord[] = [
        { id: 1, payload: { name: 'Alice' } },
        { id: 2, payload: { name: 'Bob' } },
      ];

      mockWasmDb.getPoints.mockReturnValue({
        result: records,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<PointRecord[]>);

      const points = db.getPoints('test', [1, 2]);

      expect(points).toHaveLength(2);
      expect(points[0]?.payload?.name).toBe('Alice');
    });

    it('should delete points', () => {
      const updateResult: UpdateResult = {
        status: 'completed',
        operation_id: 0,
      };

      mockWasmDb.deletePoints.mockReturnValue({
        result: updateResult,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<UpdateResult>);

      const deleted = db.deletePoints('test', [2]);

      expect(deleted).toBe(1);
    });
  });

  describe('Search', () => {
    it('should find nearest neighbors', () => {
      const searchResults: SearchResult[] = [
        { id: 1, score: 0.95, payload: { label: 'x' } },
        { id: 2, score: 0.80, payload: { label: 'y' } },
      ];

      mockWasmDb.search.mockReturnValue({
        result: searchResults,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<SearchResult[]>);

      const results = db.search('test', [1, 0.1, 0, 0], 2);

      expect(results).toHaveLength(2);
      expect(results[0].id).toBe(1);
      expect(results[0].score).toBe(0.95);
    });

    it('should search with options', () => {
      mockWasmDb.search.mockReturnValue({
        result: [],
        status: 'ok',
        time: 0.001,
      } as ApiResponse<SearchResult[]>);

      db.search('test', [1, 0, 0, 0], 5, {
        with_payload: true,
        with_vector: true,
        score_threshold: 0.5,
      });

      expect(mockWasmDb.search).toHaveBeenCalledWith(
        'test',
        expect.any(Float32Array),
        5,
        { with_payload: true, with_vector: true, score_threshold: 0.5 }
      );
    });

    it('should scroll through points', () => {
      const scrollResult: ScrollResult = {
        points: [{ id: 1 }, { id: 2 }],
        next_page_offset: 3,
      };

      mockWasmDb.scroll.mockReturnValue({
        result: scrollResult,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<ScrollResult>);

      const result = db.scroll('test', { limit: 2 });

      expect(result.points).toHaveLength(2);
      expect(result.next_page_offset).toBe(3);
    });
  });

  describe('Graph', () => {
    it('should add edges', () => {
      mockWasmDb.addEdge.mockReturnValue(undefined);

      expect(() => {
        db.addEdge('test', 1, 2, 'KNOWS');
        db.addEdge('test', 2, 3, 'KNOWS', 0.5);
      }).not.toThrow();

      expect(mockWasmDb.addEdge).toHaveBeenCalledTimes(2);
    });

    it('should traverse graph', () => {
      const traversalResult: TraversalResult = {
        nodes: [
          { id: 1, depth: 0 },
          { id: 2, depth: 1 },
          { id: 3, depth: 2 },
        ],
        edges: [
          { from: 1, to: 2, relation: 'KNOWS', weight: 1.0 },
          { from: 2, to: 3, relation: 'KNOWS', weight: 1.0 },
        ],
      };

      mockWasmDb.traverse.mockReturnValue({
        result: traversalResult,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<TraversalResult>);

      const result = db.traverse('test', 1, 2);

      expect(result.nodes).toHaveLength(3);
      expect(result.edges).toHaveLength(2);
    });

    it('should traverse with relation filter', () => {
      mockWasmDb.traverse.mockReturnValue({
        result: { nodes: [], edges: [] },
        status: 'ok',
        time: 0.001,
      } as ApiResponse<TraversalResult>);

      db.traverse('test', 1, 3, ['KNOWS', 'WORKS_WITH']);

      expect(mockWasmDb.traverse).toHaveBeenCalledWith(
        'test',
        BigInt(1),
        3,
        ['KNOWS', 'WORKS_WITH']
      );
    });
  });

  describe('Cypher', () => {
    it('should execute Cypher queries', () => {
      const cypherResult: CypherResult = {
        columns: ['n.name'],
        rows: [['Alice'], ['Bob'], ['Charlie']],
      };

      mockWasmDb.query.mockReturnValue({
        result: cypherResult,
        status: 'ok',
        time: 0.001,
      } as ApiResponse<CypherResult>);

      const result = db.query('test', 'MATCH (n) RETURN n.name LIMIT 10');

      expect(result.columns).toContain('n.name');
      expect(result.rows.length).toBe(3);
    });

    it('should execute parameterized Cypher queries', () => {
      mockWasmDb.query.mockReturnValue({
        result: { columns: ['n'], rows: [] },
        status: 'ok',
        time: 0.001,
      } as ApiResponse<CypherResult>);

      db.query('test', 'MATCH (n) WHERE n.age > $min_age RETURN n', { min_age: 25 });

      expect(mockWasmDb.query).toHaveBeenCalledWith(
        'test',
        'MATCH (n) WHERE n.age > $min_age RETURN n',
        { min_age: 25 }
      );
    });
  });

  describe('Multiple instances', () => {
    it('should support multiple database instances', async () => {
      const db2 = await LatticeDB.init();

      expect(db2).toBeInstanceOf(LatticeDB);
      expect(db2).not.toBe(db);
    });
  });
});

describe('Type exports', () => {
  it('should export all required types', async () => {
    // This test verifies that all types are properly exported
    const types = await import('../types');

    // Verify type exports exist (these are compile-time checks)
    expect(types).toBeDefined();
  });
});
