# LatticeDB Server

High-performance Vector + Graph database REST API server. Qdrant-compatible with graph extensions.

## Quick Start

```bash
# Run with default settings (port 6333)
cargo run -p lattice-server

# Run with custom address
cargo run -p lattice-server -- 127.0.0.1:8080

# Run with OpenAPI/Swagger UI
cargo run -p lattice-server --features openapi
# Visit http://localhost:6333/docs
```

## API Endpoints

### Collections

#### List Collections
```http
GET /collections
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "collections": [
      { "name": "my_collection" }
    ]
  }
}
```

#### Create Collection
```http
PUT /collections/{name}
Content-Type: application/json

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

**Parameters:**
- `vectors.size` - Vector dimensionality (required)
- `vectors.distance` - Distance metric: `Cosine`, `Euclid`, or `Dot` (required)
- `hnsw_config.m` - Max connections per node (optional, default: 16)
- `hnsw_config.ef_construct` - Construction-time search queue size (optional, default: 100)

**Response:**
```json
{
  "status": "ok",
  "result": { "result": true }
}
```

#### Get Collection Info
```http
GET /collections/{name}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "status": "green",
    "vectors_count": 1000,
    "points_count": 1000,
    "config": {
      "params": {
        "vectors": { "size": 128, "distance": "Cosine" }
      },
      "hnsw_config": { "m": 16, "ef_construct": 200 }
    }
  }
}
```

#### Delete Collection
```http
DELETE /collections/{name}
```

**Response:**
```json
{
  "status": "ok",
  "result": { "result": true }
}
```

---

### Points

#### Upsert Points
```http
PUT /collections/{name}/points
Content-Type: application/json

{
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3, ...],
      "payload": { "city": "London", "price": 100 }
    },
    {
      "id": 2,
      "vector": [0.4, 0.5, 0.6, ...],
      "payload": { "city": "Paris", "price": 200 }
    }
  ]
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "operation_id": 0,
    "status": "completed"
  }
}
```

#### Get Points by IDs
```http
POST /collections/{name}/points
Content-Type: application/json

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
    { "id": 1, "payload": { "city": "London" } },
    { "id": 2, "payload": { "city": "Paris" } }
  ]
}
```

#### Delete Points
```http
POST /collections/{name}/points/delete
Content-Type: application/json

{
  "points": [1, 2, 3]
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "operation_id": 0,
    "status": "completed"
  }
}
```

---

### Search

#### Vector Search
```http
POST /collections/{name}/points/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "limit": 10,
  "with_payload": true,
  "with_vector": false,
  "score_threshold": 0.5,
  "params": {
    "ef": 128
  }
}
```

**Parameters:**
- `vector` - Query vector (required)
- `limit` - Number of results (required)
- `with_payload` - Include payload in results (default: true)
- `with_vector` - Include vector in results (default: false)
- `score_threshold` - Filter results by score (optional)
- `params.ef` - Search queue size (optional, overrides collection default)

**Response:**
```json
{
  "status": "ok",
  "result": [
    { "id": 1, "score": 0.95, "payload": { "city": "London" } },
    { "id": 5, "score": 0.89, "payload": { "city": "Berlin" } }
  ]
}
```

#### Scroll Points
```http
POST /collections/{name}/points/scroll
Content-Type: application/json

{
  "limit": 100,
  "offset": null,
  "with_payload": true,
  "with_vector": false
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "points": [
      { "id": 1, "payload": { "city": "London" } },
      { "id": 2, "payload": { "city": "Paris" } }
    ],
    "next_page_offset": 100
  }
}
```

---

### Graph Extensions (LatticeDB)

#### Add Edge
```http
POST /collections/{name}/graph/edges
Content-Type: application/json

{
  "from_id": 1,
  "to_id": 2,
  "relation": "similar_to",
  "weight": 0.95
}
```

**Response:**
```json
{
  "status": "ok",
  "result": { "status": "created" }
}
```

#### Traverse Graph
```http
POST /collections/{name}/graph/traverse
Content-Type: application/json

{
  "start_id": 1,
  "max_depth": 3,
  "relations": ["similar_to", "related_to"]
}
```

**Parameters:**
- `start_id` - Starting point ID (required)
- `max_depth` - Maximum traversal depth (required)
- `relations` - Filter by relation types (optional, null = all relations)

**Response:**
```json
{
  "status": "ok",
  "result": {
    "visited": [2, 5, 8],
    "edges": [
      { "from_id": 1, "to_id": 2, "relation": "similar_to", "weight": 0.95 }
    ],
    "max_depth_reached": 2
  }
}
```

#### Cypher Query
```http
POST /collections/{name}/graph/query
Content-Type: application/json

{
  "query": "MATCH (n:Person) WHERE n.age > 25 RETURN n.name",
  "params": { "min_age": 25 }
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "columns": ["n.name"],
    "rows": [["Alice"], ["Bob"]],
    "stats": { "nodes_scanned": 100, "rows_returned": 2 }
  }
}
```

---

### Import/Export

Binary import/export for collection backup and migration.

#### Export Collection
```http
GET /collections/{name}/export
```

**Response:**
- Content-Type: `application/octet-stream`
- Headers: `X-Lattice-Format-Version`, `X-Lattice-Point-Count`, `X-Lattice-Dimension`
- Body: Binary data (rkyv serialized)

#### Import Collection
```http
POST /collections/{name}/import?mode={mode}
```

**Query Parameters:**
- `mode` (required): `create`, `replace`, or `merge`

**Import Modes:**
- `create` - Fail if collection exists (409)
- `replace` - Drop existing, create new
- `merge` - Add points to existing (skip duplicates)

**Request:**
- Content-Type: `application/octet-stream`
- Body: Binary data from export

**Response:**
```json
{
  "status": "ok",
  "result": {
    "points_imported": 1000,
    "points_skipped": 0,
    "dimension": 128,
    "mode": "create"
  }
}
```

**cURL Examples:**
```bash
# Export
curl http://localhost:6333/collections/test/export -o backup.lattice

# Import (create)
curl -X POST "http://localhost:6333/collections/new/import?mode=create" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @backup.lattice

# Import (merge)
curl -X POST "http://localhost:6333/collections/test/import?mode=merge" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @backup.lattice
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "status": "error",
  "error": "Collection 'test' not found"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (invalid JSON, validation error)
- `404` - Not Found (collection or point not found)
- `500` - Internal Server Error

---

## Feature Flags

| Feature | Description |
|---------|-------------|
| `native` | Enable native HTTP server (default) |
| `wasm` | Enable WASM/Service Worker transport |
| `openapi` | Enable OpenAPI/Swagger UI at `/docs` |

```bash
# Build with specific features
cargo build -p lattice-server --features "native,openapi"
```

---

## Qdrant Compatibility

This API is designed to be compatible with the [Qdrant](https://qdrant.tech/) vector database API. Most Qdrant client libraries should work with LatticeDB with minimal changes.

**Supported Qdrant endpoints:**
- Collection CRUD
- Point upsert/get/delete
- Vector search with scoring
- Scroll pagination

**LatticeDB Extensions:**
- Graph edges between points
- Graph traversal queries
- Cypher query language
- Binary import/export

---

## Security

### Authentication

Enable via environment variables:

```bash
# API Key authentication
LATTICE_API_KEYS=key1,key2,key3

# Bearer token authentication
LATTICE_BEARER_TOKENS=token1,token2
```

**Usage:**
```bash
curl -H "Authorization: ApiKey your-key" http://localhost:6333/collections
curl -H "Authorization: Bearer your-token" http://localhost:6333/collections
```

### Rate Limiting

Enable rate limiting for production:

```bash
LATTICE_RATE_LIMIT=1  # Any value enables it
```

**Defaults:** 100 requests/second, burst capacity 200

**Response Headers:**
- `X-RateLimit-Limit`: Requests per second
- `X-RateLimit-Remaining`: Available tokens
- `X-RateLimit-Reset`: Reset window (1s)
