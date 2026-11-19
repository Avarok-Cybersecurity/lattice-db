use super::*;
use crate::types::collection::{Distance, HnswConfig, VectorConfig, VectorStorage};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const OLD_CONFIG: &str = r#"{"name":"old","vectors":{"size":3,"distance":"cosine"},"hnsw":{"m":16,"m0":32,"ml":0.36,"ef":100,"ef_construction":200},"relations":{"links":0},"durability":"ephemeral"}"#;

fn hnsw() -> HnswConfig {
    HnswConfig {
        m: 16,
        m0: 32,
        ml: HnswConfig::recommended_ml(16),
        ef: 100,
        ef_construction: 200,
    }
}

fn points(storage: VectorStorage) -> Vec<Point> {
    (1..=5u64)
        .map(|id| {
            let mut v = vec![id as f32 / 3.0, 0.25, -1.0 / id as f32];
            storage.round_in_place(&mut v);
            Point::new_vector(id, v).with_field("title", format!("\"doc {id}\"").into_bytes())
        })
        .collect()
}

#[test]
fn format_two_round_trips_points_and_sections() {
    for storage in [VectorStorage::F32, VectorStorage::F16, VectorStorage::Int8] {
        let config = CollectionConfig::new(
            "c",
            VectorConfig::new(3, Distance::Cosine).with_storage(storage),
            hnsw(),
        );
        let pts = points(storage);
        let bytes = encode(&config, pts.iter(), None, None).unwrap();
        assert_eq!(&bytes[..4], b"LTDB");
        let snap = decode(&bytes).unwrap();
        assert_eq!(snap.format, SNAPSHOT_FORMAT);
        assert_eq!(
            snap.config.vectors.as_ref().map(|v| v.storage),
            Some(storage)
        );
        for (a, b) in snap.points.iter().zip(&pts) {
            assert_eq!((a.id, &a.vector, &a.payload), (b.id, &b.vector, &b.payload));
        }
    }
}

#[test]
fn compact_storage_shrinks_the_snapshot() {
    let size = |storage: VectorStorage| {
        let config = CollectionConfig::new(
            "c",
            VectorConfig::new(3, Distance::Cosine).with_storage(storage),
            hnsw(),
        );
        let pts: Vec<Point> = (0..200u64)
            .map(|i| Point::new_vector(i, vec![i as f32; 3]))
            .collect();
        encode(&config, pts.iter(), None, None).unwrap().len()
    };
    assert!(size(VectorStorage::F16) < size(VectorStorage::F32));
    assert!(size(VectorStorage::Int8) < size(VectorStorage::F16));
}

#[test]
fn format_one_still_loads_with_pre_0_6_defaults() {
    let pts = points(VectorStorage::F32);
    let snap = decode(&legacy::encode(OLD_CONFIG.as_bytes(), &pts)).unwrap();
    assert_eq!(snap.format, LEGACY_SNAPSHOT_FORMAT);
    assert!(snap.hnsw.is_none() && snap.text.is_none());
    let vectors = snap.config.vectors.expect("old configs have vectors");
    assert_eq!(
        vectors.index,
        crate::types::collection::VectorIndexKind::Hnsw
    );
    assert_eq!(vectors.storage, VectorStorage::F32);
    assert_eq!(snap.points.len(), 5);
    assert_eq!(snap.points[2].vector, pts[2].vector);
}

#[test]
fn newer_and_corrupt_snapshots_fail_clearly() {
    let config = CollectionConfig::new("c", VectorConfig::new(3, Distance::Cosine), hnsw());
    let mut bytes = encode(&config, points(VectorStorage::F32).iter(), None, None).unwrap();
    let good = bytes.clone();
    bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
    let err = decode(&bytes).unwrap_err().to_string();
    assert!(err.contains("format 3") && err.contains("newer"), "{err}");
    assert!(decode(&good[..good.len() - 1]).is_err());
    assert!(decode(b"LTD").is_err());
    assert!(decode(&good[..20]).is_err());
}

#[test]
fn collections_without_vectors_round_trip() {
    let lexical = CollectionConfig::without_vectors("c", hnsw());
    let pts: Vec<Point> = (1..=3u64)
        .map(|id| Point::new_vector(id, Vec::new()))
        .collect();
    let snap = decode(&encode(&lexical, pts.iter(), None, None).unwrap()).unwrap();
    assert!(snap.config.vectors.is_none());
    assert_eq!(snap.points.len(), 3);
    assert!(snap.points.iter().all(|p| p.vector.is_empty()));
}
