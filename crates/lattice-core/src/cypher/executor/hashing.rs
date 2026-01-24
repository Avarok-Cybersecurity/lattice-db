//! Hashing utilities for row deduplication

use crate::types::value::CypherValue;
use super::QueryExecutor;

impl QueryExecutor {
    /// Compare two rows for equality
    #[inline]
    pub(crate) fn rows_equal(&self, a: &[CypherValue], b: &[CypherValue]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(va, vb)| va == vb)
    }

    /// Hash a row for deduplication using FNV-1a algorithm
    pub(crate) fn hash_row(&self, row: &[CypherValue]) -> u64 {
        // FNV-1a constants
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for value in row {
            hash = self.hash_value(value, hash, FNV_PRIME);
        }
        hash
    }

    /// Hash a single CypherValue
    pub(crate) fn hash_value(&self, value: &CypherValue, mut hash: u64, prime: u64) -> u64 {
        // Hash type discriminant using a portable approach
        // Discriminant size varies between platforms (32-bit on WASM, 64-bit on native)
        let type_tag: u8 = match value {
            CypherValue::Null => 0,
            CypherValue::Bool(_) => 1,
            CypherValue::Int(_) => 2,
            CypherValue::Float(_) => 3,
            CypherValue::String(_) => 4,
            CypherValue::Bytes(_) => 5,
            CypherValue::Date { .. } => 6,
            CypherValue::Time { .. } => 7,
            CypherValue::DateTime { .. } => 8,
            CypherValue::Duration { .. } => 9,
            CypherValue::Point2D { .. } => 10,
            CypherValue::Point3D { .. } => 11,
            CypherValue::List(_) => 12,
            CypherValue::Map(_) => 13,
            CypherValue::NodeRef(_) => 14,
            CypherValue::RelationshipRef(_) => 15,
            CypherValue::Path(_) => 16,
        };
        hash ^= type_tag as u64;
        hash = hash.wrapping_mul(prime);

        // Hash value content
        match value {
            CypherValue::Null => hash,
            CypherValue::Bool(b) => {
                hash ^= *b as u64;
                hash.wrapping_mul(prime)
            }
            CypherValue::Int(i) => {
                for byte in i.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::Float(f) => {
                for byte in f.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::String(s) => {
                for byte in s.as_bytes() {
                    hash ^= *byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::NodeRef(id) | CypherValue::RelationshipRef(id) => {
                for byte in id.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::List(items) => {
                for item in items {
                    hash = self.hash_value(item, hash, prime);
                }
                hash
            }
            CypherValue::Map(entries) => {
                for (k, v) in entries {
                    for byte in k.as_bytes() {
                        hash ^= *byte as u64;
                        hash = hash.wrapping_mul(prime);
                    }
                    hash = self.hash_value(v, hash, prime);
                }
                hash
            }
            _ => hash, // Other types use just the discriminant
        }
    }
}
