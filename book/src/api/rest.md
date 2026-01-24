# REST API

LatticeDB provides a **Qdrant-compatible REST API** for vector operations plus extensions for graph queries. This chapter documents all available endpoints.

## Base URL

```
http://localhost:6333
```

Default port is `6333` (same as Qdrant for compatibility).

## Collections

### Create Collection

```http
PUT /collections/{collection_name}
```

**Request Body:**
```json
{
  "vectors": {
    "size": 128,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  }
}
```

**Distance Options:** `Cosine`, `Euclid`, `Dot`

**Response:**
```json
{
  "status": "ok",
  "time": 0.001
}
```

### Get Collection Info

```http
GET /collections/{collection_name}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "name": "my_collection",
    "vectors_count": 10000,
    "config": {
      "vectors": {
        "size": 128,
        "distance": "Cosine"
      },
      "hnsw_config": {
        "m": 16,
        "ef_construct": 200
      }
    }
  }
}
```

### Delete Collection

```http
DELETE /collections/{collection_name}
```

**Response:**
```json
{
  "status": "ok",
  "time": 0.002
}
```

### List Collections

```http
GET /collections
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "collections": [
      {"name": "collection_1"},
      {"name": "collection_2"}
    ]
  }
}
```

## Import/Export

Binary import/export for collection backup and migration. Uses rkyv zero-copy serialization for efficient transfer.

### Export Collection

```http
GET /collections/{collection_name}/export
```

**Response Headers:**

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/octet-stream` |
| `X-Lattice-Format-Version` | Binary format version (currently `1`) |
| `X-Lattice-Point-Count` | Number of points in collection |
| `X-Lattice-Dimension` | Vector dimension |
| `Content-Disposition` | `attachment; filename="{name}.lattice"` |

**Response Body:** Binary data (rkyv serialized collection)

### Import Collection

```http
POST /collections/{collection_name}/import?mode={mode}
```

**Query Parameters:**

| Parameter | Required | Values | Description |
|-----------|----------|--------|-------------|
| `mode` | Yes | `create`, `replace`, `merge` | Import behavior |

**Import Modes:**
- `create`: Create new collection (fails with 409 if exists)
- `replace`: Drop existing collection and create new
- `merge`: Add points to existing collection (skips duplicates)

**Request:**
- `Content-Type: application/octet-stream`
- Body: Binary data from export

**Response:**
```json
{
  "status": "ok",
  "result": {
    "points_imported": 1000,
    "points_skipped": 50,
    "dimension": 128,
    "mode": "merge"
  }
}
```

**Error Codes:**
- `400`: Invalid mode, corrupted data, or dimension mismatch (merge mode)
- `404`: Collection not found (merge mode only)
- `409`: Collection already exists (create mode)
- `413`: Payload too large (>1GB limit)

**cURL Examples:**

```bash
# Export collection to file
curl http://localhost:6333/collections/docs/export -o backup.lattice

# Import as new collection (create mode)
curl -X POST "http://localhost:6333/collections/new_docs/import?mode=create" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @backup.lattice

# Merge into existing collection
curl -X POST "http://localhost:6333/collections/docs/import?mode=merge" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @backup.lattice

# Replace existing collection
curl -X POST "http://localhost:6333/collections/docs/import?mode=replace" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @backup.lattice
```

## Points (Vectors)

### Upsert Points

```http
PUT /collections/{collection_name}/points
```

**Request Body:**
```json
{
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3, ...],
      "payload": {
        "title": "Document 1",
        "category": "tech"
      }
    },
    {
      "id": 2,
      "vector": [0.4, 0.5, 0.6, ...],
      "payload": {
        "title": "Document 2",
        "category": "science"
      }
    }
  ]
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "operation_id": 123,
    "status": "completed"
  }
}
```

### Get Points

Retrieve points by their IDs (batch operation).

```http
POST /collections/{collection_name}/points
```

**Request Body:**
```json
{
  "ids": [1, 2, 3],
  "with_payload": true,
  "with_vector": false
}
```

**Response:**
```json
{
  "status": "ok",
  "result": [
    {
      "id": 1,
      "payload": {
        "title": "Document 1",
        "category": "tech"
      }
    },
    {
      "id": 2,
      "payload": {
        "title": "Document 2",
        "category": "science"
      }
    }
  ]
}
```

### Delete Points

```http
POST /collections/{collection_name}/points/delete
```

**Request Body:**
```json
{
  "points": [1, 2, 3]
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "operation_id": 124,
    "status": "completed"
  }
}
```

## Search

### Vector Search

```http
POST /collections/{collection_name}/points/query
```

**Request Body:**
```json
{
  "query": [0.1, 0.2, 0.3, ...],
  "limit": 10,
  "ef": 100,
  "with_payload": true,
  "with_vector": false
}
```

**Parameters:**
- `query`: Query vector (required)
- `limit`: Number of results (default: 10)
- `ef`: Search quality parameter (default: 100)
- `with_payload`: Include payload in results (default: true)
- `with_vector`: Include vector in results (default: false)

**Response:**
```json
{
  "status": "ok",
  "result": [
    {
      "id": 42,
      "score": 0.95,
      "payload": {
        "title": "Most Similar Document"
      }
    },
    {
      "id": 17,
      "score": 0.89,
      "payload": {
        "title": "Second Most Similar"
      }
    }
  ],
  "time": 0.001
}
```

### Filtered Search

```http
POST /collections/{collection_name}/points/query
```

**Request Body:**
```json
{
  "query": [0.1, 0.2, 0.3, ...],
  "limit": 10,
  "filter": {
    "must": [
      {
        "key": "category",
        "match": {"value": "tech"}
      }
    ],
    "must_not": [
      {
        "key": "archived",
        "match": {"value": true}
      }
    ]
  }
}
```

### Scroll (Pagination)

```http
POST /collections/{collection_name}/points/scroll
```

**Request Body:**
```json
{
  "limit": 100,
  "offset": 0,
  "with_payload": true,
  "with_vector": false,
  "filter": {
    "must": [
      {
        "key": "category",
        "match": {"value": "tech"}
      }
    ]
  }
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "points": [...],
    "next_page_offset": 100
  }
}
```

### Batch Search

Execute multiple search queries in a single request for better efficiency.

```http
POST /collections/{collection_name}/points/search/batch
```

**Request Body:**
```json
{
  "searches": [
    {
      "vector": [0.1, 0.2, 0.3, ...],
      "limit": 10,
      "with_payload": true
    },
    {
      "vector": [0.4, 0.5, 0.6, ...],
      "limit": 5,
      "params": { "ef": 200 }
    }
  ]
}
```

**Response:**
```json
{
  "status": "ok",
  "result": [
    [
      { "id": 42, "score": 0.95, "payload": {...} },
      { "id": 17, "score": 0.89, "payload": {...} }
    ],
    [
      { "id": 8, "score": 0.91, "payload": {...} }
    ]
  ]
}
```

## Graph Operations

### Add Edge

```http
POST /collections/{collection_name}/graph/edges
```

**Request Body:**
```json
{
  "source_id": 1,
  "target_id": 2,
  "weight": 0.9,
  "relation": "REFERENCES"
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "created": true
  }
}
```

### Get Neighbors

```http
GET /collections/{collection_name}/graph/neighbors/{point_id}
```

**Query Parameters:**
- `relation`: Filter by relation type (optional)
- `direction`: `outgoing` (default), `incoming`, or `both`

**Response:**
```json
{
  "status": "ok",
  "result": {
    "neighbors": [
      {
        "id": 2,
        "weight": 0.9,
        "relation": "REFERENCES"
      },
      {
        "id": 3,
        "weight": 0.7,
        "relation": "CITES"
      }
    ]
  }
}
```

### Traverse Graph

Perform BFS/DFS traversal from a starting point.

```http
POST /collections/{collection_name}/graph/traverse
```

**Request Body:**
```json
{
  "start_id": 1,
  "max_depth": 3,
  "relations": ["KNOWS", "REFERENCES"]
}
```

**Parameters:**
- `start_id`: Starting point ID (required)
- `max_depth`: Maximum traversal depth (required, max: 100)
- `relations`: Filter by relation types (optional, null = all relations)

**Response:**
```json
{
  "status": "ok",
  "result": {
    "visited": [2, 5, 8, 12],
    "edges": [
      {"from_id": 1, "to_id": 2, "relation": "KNOWS", "weight": 0.95},
      {"from_id": 2, "to_id": 5, "relation": "KNOWS", "weight": 0.87}
    ],
    "max_depth_reached": 2
  }
}
```

### Cypher Query

```http
POST /collections/{collection_name}/graph/query
```

**Request Body:**
```json
{
  "query": "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name",
  "params": {
    "min_age": 25
  }
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "columns": ["n.name"],
    "rows": [
      ["Alice"],
      ["Bob"],
      ["Charlie"]
    ],
    "stats": {
      "nodes_scanned": 100,
      "rows_returned": 3,
      "execution_time_ms": 5
    }
  }
}
```

## Error Responses

All errors return a consistent format:

```json
{
  "status": "error",
  "message": "Collection 'xyz' not found",
  "code": 404
}
```

### Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Collection/point doesn't exist |
| 409 | Conflict - Collection already exists |
| 500 | Internal Server Error |

## Headers

### Request Headers

| Header | Value | Description |
|--------|-------|-------------|
| `Content-Type` | `application/json` | Required for POST/PUT |
| `Accept` | `application/json` | Optional |

### Response Headers

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `Server-Timing` | RFC 6797 timing: `body;dur=X, handler;dur=Y, total;dur=Z` (microseconds) |

## Authentication

LatticeDB supports API key and Bearer token authentication. By default, authentication is disabled.

### Enabling Authentication

Set one or both environment variables:

```bash
# API Key authentication
LATTICE_API_KEYS=key1,key2,key3

# Bearer token authentication
LATTICE_BEARER_TOKENS=token1,token2
```

### Making Authenticated Requests

```bash
# Using API Key
curl -H "Authorization: ApiKey your-api-key" \
  http://localhost:6333/collections

# Using Bearer Token
curl -H "Authorization: Bearer your-token" \
  http://localhost:6333/collections
```

### Public Endpoints

These endpoints do not require authentication:
- `GET /` - Root endpoint
- `GET /health`, `/healthz` - Health check
- `GET /ready`, `/readyz` - Readiness check

## Rate Limiting

By default, LatticeDB has no rate limiting. Enable it for production:

```bash
LATTICE_RATE_LIMIT=1  # Any value enables rate limiting
```

**Default limits:** 100 requests/second, burst capacity 200

### Rate Limit Headers

When rate limiting is enabled, responses include:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Requests allowed per second |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Seconds until limit resets |

**429 Too Many Requests** is returned when limits are exceeded.

## TLS/HTTPS

Enable TLS for encrypted connections:

```bash
LATTICE_TLS_CERT=/path/to/cert.pem
LATTICE_TLS_KEY=/path/to/key.pem
```

Requires building with `--features tls`.

## CORS

CORS is enabled by default for browser access:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `LATTICE_HOST` | Server bind address | `0.0.0.0` |
| `LATTICE_PORT` | Server port | `6333` |
| `LATTICE_API_KEYS` | Comma-separated API keys for authentication | (disabled) |
| `LATTICE_BEARER_TOKENS` | Comma-separated Bearer tokens for authentication | (disabled) |
| `LATTICE_RATE_LIMIT` | Enable rate limiting (any value) | (disabled) |
| `LATTICE_TLS_CERT` | Path to TLS certificate file | (disabled) |
| `LATTICE_TLS_KEY` | Path to TLS private key file | (disabled) |
| `LATTICE_DATA_DIR` | Data persistence directory | `./data` |
| `LATTICE_LOG_LEVEL` | Logging verbosity (`error`, `warn`, `info`, `debug`, `trace`) | `info` |

## cURL Examples

### Create and Populate

```bash
# Create collection
curl -X PUT http://localhost:6333/collections/docs \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 128, "distance": "Cosine"}}'

# Add points
curl -X PUT http://localhost:6333/collections/docs/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {"id": 1, "vector": [0.1, 0.2, ...], "payload": {"title": "Doc 1"}}
    ]
  }'

# Search
curl -X POST http://localhost:6333/collections/docs/points/query \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, 0.2, ...], "limit": 5}'
```

### Graph Operations

```bash
# Add edge
curl -X POST http://localhost:6333/collections/docs/graph/edges \
  -H "Content-Type: application/json" \
  -d '{"source_id": 1, "target_id": 2, "weight": 0.9, "relation": "REFS"}'

# Cypher query
curl -X POST http://localhost:6333/collections/docs/graph/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n LIMIT 10"}'
```

## Next Steps

- [Rust API](./rust.md) - Native Rust integration
- [TypeScript API](./typescript.md) - Browser/Node.js client
