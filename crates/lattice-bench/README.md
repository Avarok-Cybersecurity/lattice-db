# LatticeDB Cypher Benchmarks

Benchmark suite comparing LatticeDB's Cypher implementation against Neo4j.

## Prerequisites

### Start Neo4j (Docker)

```bash
docker run -d \
  --name neo4j-bench \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/benchmarkpassword \
  neo4j:5
```

Wait for Neo4j to start (check logs):
```bash
docker logs -f neo4j-bench
```

### Verify Neo4j is Running

Open http://localhost:7474 in your browser or:
```bash
curl -I http://localhost:7474
```

## Running Benchmarks

### Full Benchmark Suite

```bash
cargo bench -p lattice-bench
```

### Specific Benchmark Group

```bash
cargo bench -p lattice-bench -- match_all
cargo bench -p lattice-bench -- node_creation
cargo bench -p lattice-bench -- match_with_filter
```

### LatticeDB Only (No Neo4j)

If Neo4j is not running, the benchmarks will automatically skip Neo4j tests and only run LatticeDB benchmarks.

## Benchmark Categories

| Category | Description |
|----------|-------------|
| `node_creation` | Single node CREATE performance |
| `match_all` | MATCH (n) RETURN n |
| `match_by_label` | MATCH (n:Label) RETURN n |
| `match_with_limit` | MATCH (n:Label) RETURN n LIMIT 10 |
| `match_with_filter` | MATCH (n) WHERE n.prop > val RETURN n |
| `complex_filter` | MATCH (n) WHERE n.a > x AND n.b = y RETURN n |
| `projection` | MATCH (n) RETURN n.prop1, n.prop2 |
| `order_by` | MATCH (n) RETURN n ORDER BY n.prop |
| `skip_limit` | MATCH (n) RETURN n SKIP x LIMIT y |

## Dataset

Benchmarks use generated "Person" nodes with properties:
- `name`: String (e.g., "Alice Smith")
- `age`: Integer (18-80)
- `city`: String (10 major US cities)
- `email`: String (generated email)

Data sizes tested: 100, 500, 1000 nodes

## Output

Results are saved to `target/criterion/` with HTML reports.

View the report:
```bash
open target/criterion/report/index.html
```

## Example Results

Typical results on a development machine (Apple M2):

```
match_all/LatticeDB/1000   time:   [45.2 µs 46.1 µs 47.0 µs]
match_all/Neo4j/1000       time:   [2.1 ms  2.2 ms  2.3 ms]
                           ↑ LatticeDB ~45x faster

match_with_filter/LatticeDB/1000  time:   [52.3 µs 53.1 µs 54.0 µs]
match_with_filter/Neo4j/1000      time:   [1.8 ms  1.9 ms  2.0 ms]
                                  ↑ LatticeDB ~35x faster
```

Note: Neo4j performance includes network latency (Bolt protocol). LatticeDB runs in-process with zero network overhead.

## Cleanup

```bash
docker stop neo4j-bench && docker rm neo4j-bench
```
