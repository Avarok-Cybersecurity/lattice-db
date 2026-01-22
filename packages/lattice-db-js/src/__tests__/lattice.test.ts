/**
 * LatticeDB TypeScript Library Tests
 *
 * Basic tests for type exports and API shape validation.
 * Full integration tests run in Rust via wasm-bindgen-test.
 *
 * Note: WASM integration tests require a browser environment.
 * These tests verify TypeScript type correctness only.
 */

import { describe, it, expect } from '@jest/globals';

describe('Type exports', () => {
  it('should export all types from types.ts', async () => {
    const types = await import('../types.js');
    expect(types).toBeDefined();
  });

  it('should have correct type structure for CollectionConfig', async () => {
    // Type-level test - if this compiles, the types are correct
    const _config: import('../types.js').CollectionConfig = {
      vectors: { size: 128, distance: 'Cosine' },
    };
    expect(_config.vectors.size).toBe(128);
  });

  it('should have correct CollectionConfig with HNSW config', async () => {
    const _config: import('../types.js').CollectionConfig = {
      vectors: { size: 128, distance: 'Euclid' },
      hnsw_config: { m: 16, ef_construct: 200 },
    };
    expect(_config.hnsw_config?.m).toBe(16);
  });

  it('should have correct type structure for Point', async () => {
    const _point: import('../types.js').Point = {
      id: 1,
      vector: [0.1, 0.2, 0.3],
      payload: { name: 'test' },
    };
    expect(_point.id).toBe(1);
    expect(_point.vector).toHaveLength(3);
  });

  it('should allow Point without payload', async () => {
    const _point: import('../types.js').Point = {
      id: 1,
      vector: [0.1, 0.2, 0.3],
    };
    expect(_point.payload).toBeUndefined();
  });

  it('should have correct type structure for PointRecord', async () => {
    const _record: import('../types.js').PointRecord = {
      id: 42,
      vector: [1, 2, 3],
      payload: { category: 'A' },
    };
    expect(_record.id).toBe(42);
  });

  it('should have correct type structure for SearchResult', async () => {
    const _result: import('../types.js').SearchResult = {
      id: 1,
      score: 0.95,
      payload: { label: 'test' },
    };
    expect(_result.score).toBe(0.95);
  });

  it('should have correct type structure for SearchOptions', async () => {
    const _opts: import('../types.js').SearchOptions = {
      with_payload: true,
      with_vector: false,
      score_threshold: 0.7,
    };
    expect(_opts.score_threshold).toBe(0.7);
  });

  it('should have correct type structure for ScrollOptions', async () => {
    const _opts: import('../types.js').ScrollOptions = {
      limit: 100,
      offset: 50,
      with_payload: true,
    };
    expect(_opts.limit).toBe(100);
  });

  it('should have correct type structure for ScrollResult', async () => {
    const _result: import('../types.js').ScrollResult = {
      points: [{ id: 1 }, { id: 2 }],
      next_page_offset: 3,
    };
    expect(_result.points).toHaveLength(2);
    expect(_result.next_page_offset).toBe(3);
  });

  it('should have correct type structure for CypherResult', async () => {
    const _result: import('../types.js').CypherResult = {
      columns: ['n.name'],
      rows: [['Alice'], ['Bob']],
    };
    expect(_result.columns).toContain('n.name');
    expect(_result.rows).toHaveLength(2);
  });

  it('should have correct type structure for TraversalNode', async () => {
    const _node: import('../types.js').TraversalNode = {
      id: 1,
      depth: 0,
      payload: { name: 'root' },
    };
    expect(_node.depth).toBe(0);
  });

  it('should have correct type structure for TraversalEdge', async () => {
    const _edge: import('../types.js').TraversalEdge = {
      from: 1,
      to: 2,
      relation: 'KNOWS',
      weight: 1.0,
    };
    expect(_edge.relation).toBe('KNOWS');
  });

  it('should have correct type structure for TraversalResult', async () => {
    const _result: import('../types.js').TraversalResult = {
      nodes: [{ id: 1, depth: 0 }],
      edges: [{ from: 1, to: 2, relation: 'KNOWS', weight: 1.0 }],
    };
    expect(_result.nodes).toHaveLength(1);
    expect(_result.edges).toHaveLength(1);
  });

  it('should have correct type structure for UpsertResult', async () => {
    const _result: import('../types.js').UpsertResult = {
      status: 'completed',
      operation_id: 0,
    };
    expect(_result.status).toBe('completed');
  });

  it('should have correct type structure for ApiResponse', async () => {
    const _response: import('../types.js').ApiResponse<string[]> = {
      status: 'ok',
      result: ['a', 'b'],
      time: 0.001,
    };
    expect(_response.result).toHaveLength(2);
    expect(_response.status).toBe('ok');
  });
});

describe('Distance metrics', () => {
  it('should accept valid distance metrics', () => {
    const metrics: import('../types.js').DistanceMetric[] = ['Cosine', 'Euclid', 'Dot'];
    expect(metrics).toHaveLength(3);
  });
});

describe('Payload type', () => {
  it('should accept any JSON-serializable payload', () => {
    const _payload: import('../types.js').Payload = {
      string: 'value',
      number: 42,
      boolean: true,
      array: [1, 2, 3],
      nested: { key: 'value' },
    };
    expect(Object.keys(_payload)).toHaveLength(5);
  });
});
