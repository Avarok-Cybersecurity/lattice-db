# ACID Mode: Future Enhancements

This document outlines advanced ACID features planned for future releases of LatticeDB, beyond the initial Phase 1 implementation (WAL, crash recovery, auto-commit transactions).

---

## 1. Multi-Operation Transactions

### Current State (Phase 1)
Each operation (upsert, delete, add_edge) is its own auto-committed transaction.

### Future Enhancement
Support explicit transaction boundaries:

```rust
// API
POST /collections/{name}/transaction/begin
POST /collections/{name}/transaction/commit
POST /collections/{name}/transaction/rollback

// Usage
let txn_id = client.begin_transaction("my_collection").await?;
client.upsert_in_txn(txn_id, points).await?;
client.add_edge_in_txn(txn_id, edge).await?;
client.commit(txn_id).await?;  // All-or-nothing
```

### Implementation Notes
- Transaction ID tracking
- Pending writes buffer per transaction
- Rollback = discard buffer
- Commit = write all to WAL atomically, then apply

### Use Cases
- Atomic batch imports
- Consistent graph + vector updates
- Application-level consistency requirements

---

## 2. MVCC (Multi-Version Concurrency Control)

### Current State (Phase 1)
Serializable isolation via write locks. Readers block writers.

### Future Enhancement
Snapshot isolation with versioned data:

```rust
// Each point/edge has version metadata
struct VersionedPoint {
    point: Point,
    created_at: u64,    // LSN when created
    deleted_at: Option<u64>,  // LSN when deleted (tombstone)
}

// Readers see consistent snapshot at their start LSN
// Writers create new versions, don't block readers
```

### Benefits
- Non-blocking reads during writes
- Consistent snapshots for long-running queries
- Time-travel queries ("show me data as of LSN X")

### Trade-offs
- Higher memory usage (multiple versions)
- Garbage collection needed for old versions
- More complex recovery

---

## 3. Distributed WAL Replication

### Current State (Phase 1)
Single-node, local WAL file.

### Future Enhancement
Replicate WAL to multiple nodes for high availability:

```
┌─────────────┐         ┌─────────────┐
│   Primary   │ ──WAL──▶│  Replica 1  │
│   (writes)  │         │  (reads)    │
└─────────────┘         └─────────────┘
       │
       └────WAL────────▶┌─────────────┐
                        │  Replica 2  │
                        │  (reads)    │
                        └─────────────┘
```

### Replication Modes

| Mode | Latency | Durability | Use Case |
|------|---------|------------|----------|
| Async | Low | Eventually consistent | High throughput |
| Semi-sync | Medium | At least 1 replica | Balanced |
| Sync | High | All replicas | Critical data |

### Implementation Notes
- WAL shipping protocol (push or pull)
- Replica lag tracking
- Automatic failover
- Conflict resolution (primary wins)

---

## 4. Incremental Checkpoints

### Current State (Phase 1)
Full snapshot on checkpoint. WAL replays from last checkpoint.

### Future Enhancement
Incremental checkpoints that only persist changed pages:

```rust
struct IncrementalCheckpoint {
    base_checkpoint_lsn: u64,
    dirty_pages: Vec<(PageId, Vec<u8>)>,
    metadata_delta: HashMap<String, Vec<u8>>,
}
```

### Benefits
- Faster checkpoint creation
- Less I/O during checkpoints
- Shorter recovery time (merge incrementals)

### Implementation
- Track dirty pages via copy-on-write or dirty flags
- Periodic merge of incrementals into full checkpoint
- Background checkpoint thread

---

## 5. WAL Compression

### Current State (Phase 1)
Uncompressed rkyv-serialized WAL entries.

### Future Enhancement
Optional compression for WAL entries:

```rust
enum WalCompression {
    None,           // Current behavior
    Lz4,            // Fast compression (default)
    Zstd,           // Better ratio, slower
    ZstdDictionary, // Best for repetitive data
}

struct CompressedWalEntry {
    compression: WalCompression,
    uncompressed_size: u32,
    data: Vec<u8>,
}
```

### Benefits
- Reduced disk I/O (faster writes)
- Smaller WAL files
- Faster replication (less network)

### Trade-offs
- CPU overhead for compression
- Slightly slower recovery (decompression)

### Recommendation
- LZ4 for latency-sensitive workloads
- Zstd for storage-constrained environments

---

## 6. Read Replicas

### Current State (Phase 1)
Single engine instance serves all reads and writes.

### Future Enhancement
Separate read-only replicas that follow primary's WAL:

```rust
// Primary
let primary = CollectionEngine::new_primary(config, storage).await?;

// Read replica (follows primary's WAL)
let replica = CollectionEngine::new_replica(
    config,
    WalFollower::connect("primary:9000").await?
).await?;

// Reads go to replica, writes go to primary
```

### Benefits
- Scale read throughput horizontally
- Geographic distribution
- Isolation of analytics queries

### Implementation
- WAL streaming protocol
- Replica lag monitoring
- Automatic promotion on primary failure

---

## 7. Point-in-Time Recovery (PITR)

### Current State (Phase 1)
Recovery to last consistent state only.

### Future Enhancement
Recovery to any point in time:

```bash
# Recover to specific LSN
lattice-recover --data-dir /data --target-lsn 12345

# Recover to timestamp
lattice-recover --data-dir /data --target-time "2024-01-15T10:30:00Z"
```

### Requirements
- Archived WAL segments (not truncated)
- LSN-to-timestamp mapping
- Base backups at regular intervals

### Use Cases
- Undo accidental deletes
- Forensic analysis
- Regulatory compliance

---

## 8. Two-Phase Commit (2PC)

### Current State (Phase 1)
Single-collection transactions only.

### Future Enhancement
Atomic transactions across multiple collections:

```rust
let txn = client.begin_distributed_txn().await?;

// Modify multiple collections atomically
txn.upsert("vectors", points).await?;
txn.upsert("metadata", meta_points).await?;
txn.add_edge("graph", edge).await?;

txn.commit().await?;  // All-or-nothing across collections
```

### Implementation
- Coordinator (transaction manager)
- Prepare phase (all participants vote)
- Commit phase (if all vote yes)
- Recovery for in-doubt transactions

---

## Implementation Priority

| Feature | Complexity | Impact | Priority |
|---------|------------|--------|----------|
| WAL Compression | Low | Medium | P1 |
| Incremental Checkpoints | Medium | High | P1 |
| Multi-Op Transactions | Medium | High | P2 |
| Read Replicas | High | High | P2 |
| MVCC | High | Medium | P3 |
| Distributed Replication | Very High | High | P3 |
| PITR | Medium | Low | P4 |
| 2PC | Very High | Low | P4 |

---

## Contributing

These features are open for community contribution. If you're interested in implementing any of these, please:

1. Open an issue to discuss the design
2. Reference this document
3. Follow the existing code patterns (SBIO, PCND, etc.)

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.
