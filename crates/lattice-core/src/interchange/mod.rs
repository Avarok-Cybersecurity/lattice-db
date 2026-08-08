//! Logical import/export — the portable, text-based counterpart to the on-disk format.
//!
//! LatticeDB persists through [`crate::storage::LatticeStorage`], which is a
//! **page** interface: a block device. Copying those pages is a *physical*
//! backup — fast, exact, and tied to the storage layout of the version that
//! wrote it. This module is the *logical* counterpart, the same split every
//! mature database ends up making:
//!
//! | | physical | logical |
//! |---|---|---|
//! | Postgres | base backup | `pg_dump` |
//! | SQLite | the `.db` file | `.dump` |
//! | Git | packfiles | `fast-export` |
//! | LatticeDB | rkyv pages | **this module** |
//!
//! A logical dump answers the questions a page dump cannot: move data between
//! versions whose page layout differs, read it without the engine that wrote
//! it, keep it in version control where a human reviews the diff, and ship
//! seed data or test fixtures as source.
//!
//! # Why this is not implemented on `LatticeStorage`
//!
//! Because that trait speaks in [`crate::storage::Page`]. A JSONL file full of
//! base64 pages is text on the surface and opaque underneath — `git diff` would
//! report "line 400 changed" and tell the reader nothing. Interchange has to
//! happen where the data still has *meaning*, which is [`crate::EngineOps`].
//!
//! # Relation ids are not portable, relation names are
//!
//! [`crate::Edge`] stores `relation_id: u16`, which is an index into the
//! collection's own `relations` map. Exporting the number would produce a file
//! that silently means something different when imported into a collection
//! whose map assigns the ids in another order — the exact failure a physical
//! dump has and a logical one must not. Edges are therefore written with their
//! **resolved relation name**, and reconnected by name on import.
//!
//! # Determinism is the whole point
//!
//! Export sorts points by id, edges by `(from, relation, to)`, and payload keys
//! lexicographically. Without that, two dumps of identical data differ on every
//! line, diffs are noise, and the format fails at the one job it exists for.
//! The tests assert byte-identical output across repeated exports.

mod jsonl;

#[cfg(test)]
#[path = "jsonl_tests.rs"]
mod jsonl_tests;

pub use jsonl::{
    export_jsonl, import_jsonl, Dump, ExportOptions, Header, VectorMode, FORMAT, FORMAT_VERSION,
};
