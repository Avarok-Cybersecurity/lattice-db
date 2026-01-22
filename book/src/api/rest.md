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

### Get Point

```http
GET /collections/{collection_name}/points/{point_id}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "id": 1,
    "vector": [0.1, 0.2, 0.3, ...],
    "payload": {
      "title": "Document 1",
      "category": "tech"
    }
  }
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
| `X-Response-Time` | Request duration in ms |

## Rate Limiting

By default, LatticeDB has no rate limiting. For production deployments, configure limits via environment variables:

```bash
LATTICE_MAX_REQUESTS_PER_SECOND=1000
LATTICE_MAX_CONCURRENT_REQUESTS=100
```

## CORS

CORS is enabled by default for browser access:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

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
