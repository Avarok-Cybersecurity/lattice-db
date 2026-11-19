//! Write hash maps in key order, so their encoding does not depend on the
//! map's hasher seed (`RandomState` differs per map and per process).
//!
//! Neither wrapper changes what is written, only the order it is written in:
//! [`SortedKeys`] archives the same `ArchivedHashMap` rkyv's own impl does,
//! and [`serialize_sorted`] emits the same JSON object, so readers of
//! existing snapshots, WAL records and configurations are unaffected.

use rkyv::collections::swiss_table::{ArchivedHashMap, HashMapResolver};
use rkyv::rancor::{Fallible, Source};
use rkyv::ser::{Allocator, Writer};
use rkyv::with::{ArchiveWith, DeserializeWith, SerializeWith};
use rkyv::{Archive, Deserialize, Place, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hash};

/// The load factor rkyv's `HashMap` impl archives with; the archived table's
/// size depends on it, so it must match for the bytes to stay compatible.
const LOAD_FACTOR: (usize, usize) = (7, 8);

/// `#[rkyv(with = SortedKeys)]`: archive a `HashMap` with its entries
/// inserted into the archived table in ascending key order.
///
/// The archived table is placed by a fixed hasher, but entries whose probe
/// sequences collide take slots in insertion order, so iterating a
/// `RandomState` map straight into it yields seed-dependent bytes.
pub struct SortedKeys;

impl<K, V, H> ArchiveWith<HashMap<K, V, H>> for SortedKeys
where
    K: Archive + Hash + Eq,
    K::Archived: Hash + Eq,
    V: Archive,
{
    type Archived = ArchivedHashMap<K::Archived, V::Archived>;
    type Resolver = HashMapResolver;

    fn resolve_with(
        field: &HashMap<K, V, H>,
        resolver: Self::Resolver,
        out: Place<Self::Archived>,
    ) {
        ArchivedHashMap::resolve_from_len(field.len(), LOAD_FACTOR, resolver, out);
    }
}

impl<K, V, H, S> SerializeWith<HashMap<K, V, H>, S> for SortedKeys
where
    K: Serialize<S> + Hash + Eq + Ord,
    K::Archived: Hash + Eq,
    V: Serialize<S>,
    S: Fallible + Writer + Allocator + ?Sized,
    S::Error: Source,
{
    fn serialize_with(
        field: &HashMap<K, V, H>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        let mut entries: Vec<(&K, &V)> = field.iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        ArchivedHashMap::<K::Archived, V::Archived>::serialize_from_iter::<_, _, _, K, V, _>(
            entries.iter().copied(),
            LOAD_FACTOR,
            serializer,
        )
    }
}

impl<K, V, H, D> DeserializeWith<ArchivedHashMap<K::Archived, V::Archived>, HashMap<K, V, H>, D>
    for SortedKeys
where
    K: Archive + Hash + Eq,
    K::Archived: Deserialize<K, D> + Hash + Eq,
    V: Archive,
    V::Archived: Deserialize<V, D>,
    D: Fallible + ?Sized,
    H: Default + BuildHasher,
{
    fn deserialize_with(
        field: &ArchivedHashMap<K::Archived, V::Archived>,
        deserializer: &mut D,
    ) -> Result<HashMap<K, V, H>, D::Error> {
        field.deserialize(deserializer)
    }
}

/// `#[serde(serialize_with = "serialize_sorted")]`: write a `HashMap` as a
/// JSON object with its keys in ascending order.
pub fn serialize_sorted<K, V, H, S>(
    map: &HashMap<K, V, H>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    K: serde::Serialize + Ord,
    V: serde::Serialize,
    S: serde::Serializer,
{
    let sorted: BTreeMap<&K, &V> = map.iter().collect();
    serde::Serialize::serialize(&sorted, serializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::rancor::Error;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[derive(Archive, Serialize, Deserialize)]
    struct Sorted(#[rkyv(with = SortedKeys)] HashMap<String, Vec<u8>>);

    fn map(keys: impl Iterator<Item = u32>) -> HashMap<String, Vec<u8>> {
        keys.map(|k| (format!("key{k}"), k.to_le_bytes().to_vec()))
            .collect()
    }

    #[test]
    fn archive_is_independent_of_the_hasher_seed() {
        let reference = rkyv::to_bytes::<Error>(&Sorted(map(0..64))).expect("archive");
        for _ in 0..8 {
            let again = rkyv::to_bytes::<Error>(&Sorted(map((0..64).rev()))).expect("archive");
            assert_eq!(again.as_slice(), reference.as_slice());
        }
        let read = rkyv::from_bytes::<Sorted, Error>(&reference).expect("read");
        assert_eq!(read.0, map(0..64));
    }

    #[test]
    fn json_keys_are_ascending() {
        let m: HashMap<String, u16> = (0..16u16).rev().map(|i| (format!("r{i:02}"), i)).collect();
        let mut json = serde_json::Serializer::new(Vec::new());
        serialize_sorted(&m, &mut json).expect("serialize");
        let expected: BTreeMap<String, u16> = m.clone().into_iter().collect();
        assert_eq!(
            json.into_inner(),
            serde_json::to_vec(&expected).expect("serialize")
        );
    }
}
