# Enterprise Error Handling Implementation Plan

## Overview

Replace all `unwrap()` calls with proper error handling using a static error enum system. This is a multi-phase implementation targeting ~40 production unwraps.

---

## Phase 1: Error Type Infrastructure

### 1.1 Modify `crates/lattice-core/src/error.rs`

Add new variants to `LatticeError`:

```rust
/// Internal error - lock poisoning, invariant violations
#[error("Internal error [{code}]: {message}")]
Internal {
    code: u32,
    message: String,
},

/// Data corruption detected
#[error("Data corruption: {context}")]
DataCorruption { context: String },
```

Add new variants to `IndexError`:

```rust
/// NaN or infinite value in vector data
#[error("Invalid vector value at dimension {dimension}: {value_type}")]
InvalidVectorValue { dimension: usize, value_type: &'static str },

/// Entry point missing (empty index)
#[error("Index has no entry point - insert points first")]
NoEntryPoint,
```

### 1.2 Modify `crates/lattice-core/src/storage.rs`

Add to `StorageError`:

```rust
/// File format corruption
#[error("Corrupted file at offset {offset}: {reason}")]
CorruptedFile { offset: usize, reason: String },

/// Feature not implemented
#[error("Feature not implemented: {feature}")]
NotImplemented { feature: &'static str },
```

### 1.3 Create `crates/lattice-core/src/sync.rs`

New module with lock extension trait:

```rust
use crate::{LatticeError, LatticeResult};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub trait LockExt<T> {
    fn read_safe(&self) -> LatticeResult<RwLockReadGuard<'_, T>>;
    fn write_safe(&self) -> LatticeResult<RwLockWriteGuard<'_, T>>;
}

impl<T> LockExt<T> for RwLock<T> {
    fn read_safe(&self) -> LatticeResult<RwLockReadGuard<'_, T>> {
        self.read().map_err(|_| LatticeError::Internal {
            code: 50001,
            message: "RwLock poisoned during read".into(),
        })
    }

    fn write_safe(&self) -> LatticeResult<RwLockWriteGuard<'_, T>> {
        self.write().map_err(|_| LatticeError::Internal {
            code: 50002,
            message: "RwLock poisoned during write".into(),
        })
    }
}

/// Compare f32 values with NaN handling (NaN sorts to end)
#[inline]
pub fn cmp_f32(a: f32, b: f32) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => std::cmp::Ordering::Equal,
        }
    })
}
```

---

## Phase 2: Lock Handling (collection.rs) - 20+ unwraps

### File: `crates/lattice-core/src/engine/collection.rs`

**Pattern replacement:**
```rust
// Before
let pts = self.points.read().unwrap();

// After
use crate::sync::LockExt;
let pts = self.points.read_safe()?;
```

**Lines to modify:** 152, 157, 167, 192-199, 223-225, 300, 321, 360, 382, 434-436, 465

**Special cases** (non-Result return types):
- `point_count()` → use `unwrap_or(0)`
- `is_empty()` → use `unwrap_or(true)`

---

## Phase 3: NaN Handling (hnsw.rs) - 7 unwraps

### File: `crates/lattice-core/src/index/hnsw.rs`

**Pattern replacement:**
```rust
// Before
sorted.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

// After
use crate::sync::cmp_f32;
sorted.sort_unstable_by(|a, b| cmp_f32(a.distance, b.distance));
```

**Lines to modify:** 545, 559, 641, 652, 766, 986

**Entry point checks (lines 173, 251, 866):**
```rust
// Before
let entry = self.layers.entry_point().unwrap();

// After
let entry = self.layers.entry_point()
    .ok_or(IndexError::NoEntryPoint)?;
```

---

## Phase 4: Mmap Parsing (mmap_vectors.rs) - 7 unwraps

### File: `crates/lattice-core/src/index/mmap_vectors.rs`

**Add helper function:**
```rust
fn read_u64_le(data: &[u8], offset: usize) -> io::Result<u64> {
    let end = offset + 8;
    if data.len() < end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("File truncated at offset {}", offset),
        ));
    }
    Ok(u64::from_le_bytes(data[offset..end].try_into().unwrap()))
}
```

**Lines to modify:** 68, 76, 84, 87, 94, 98

---

## Phase 5: Todo Removal

### File: `crates/lattice-storage/src/disk.rs`

Replace 6 `todo!()` calls with:
```rust
Err(StorageError::NotImplemented { feature: "DiskStorage::method_name" })
```

**Lines:** 34, 38, 42, 50, 54, 62

### File: `crates/lattice-server/src/service_worker.rs`

Line 67: Replace panic with compile_error or abort.

---

## Phase 6: Update Exports

### File: `crates/lattice-core/src/lib.rs`

Add:
```rust
pub mod sync;
pub use sync::{LockExt, cmp_f32};
```

---

## Verification

```bash
# After each phase:
cargo check -p lattice-core
cargo test -p lattice-core
cargo clippy -p lattice-core -- -D warnings

# Final verification:
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# Verify no unwrap in production code:
grep -r "\.unwrap()" crates/lattice-core/src --include="*.rs" | grep -v "#\[test\]" | grep -v "mod tests"
```

---

## Files to Modify (in order)

1. `crates/lattice-core/src/error.rs` - Add Internal, DataCorruption, IndexError variants
2. `crates/lattice-core/src/storage.rs` - Add CorruptedFile, NotImplemented
3. `crates/lattice-core/src/sync.rs` - NEW: LockExt trait, cmp_f32
4. `crates/lattice-core/src/lib.rs` - Export sync module
5. `crates/lattice-core/src/engine/collection.rs` - Replace lock unwraps
6. `crates/lattice-core/src/engine/async_indexer.rs` - Replace lock unwraps
7. `crates/lattice-core/src/index/hnsw.rs` - Replace NaN unwraps
8. `crates/lattice-core/src/index/mmap_vectors.rs` - Replace parsing unwraps
9. `crates/lattice-storage/src/disk.rs` - Replace todo!()
10. `crates/lattice-server/src/service_worker.rs` - Replace panic

---

## Error Code Scheme

| Range | Category |
|-------|----------|
| 50001 | Lock read poison |
| 50002 | Lock write poison |
| 50003 | Mutex poison |
| 60001 | File magic mismatch |
| 60002 | File truncated |
| 60003 | Version mismatch |

---

## Post-Implementation

Update `SECURITY_RECOMMENDATIONS.md` to mark completed:
- [x] **167 `unwrap()` calls in lattice-core** - Replaced with proper error handling
- [x] **5 `panic!()` calls** - Converted to errors (test panics kept)
