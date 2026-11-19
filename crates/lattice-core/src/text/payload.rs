//! Keeps a [`LexicalIndex`] in step with the points of a collection.
//!
//! The engines call [`PayloadTextIndex::upsert_point`] / [`remove_point`]
//! from their single upsert and delete paths, which are also the paths used by
//! snapshot restore, WAL replay and JSONL import — so a loaded corpus always
//! has its index rebuilt deterministically from the stored payloads.
//!
//! [`remove_point`]: PayloadTextIndex::remove_point

use super::bm25::{Bm25Params, LexicalIndex};
use super::config::TextIndexConfig;
use crate::types::point::{Point, PointId};
use std::collections::HashMap;

/// A BM25 index bound to one payload field.
#[derive(Debug, Clone)]
pub struct PayloadTextIndex {
    field: String,
    index: LexicalIndex,
}

impl PayloadTextIndex {
    pub fn new(config: &TextIndexConfig) -> Self {
        Self {
            field: config.field.clone(),
            index: LexicalIndex::new(Bm25Params {
                k1: config.k1,
                b: config.b,
            }),
        }
    }

    /// Build the index for a collection, if it configures one.
    pub fn for_config(config: Option<&TextIndexConfig>) -> Option<Self> {
        config.map(Self::new)
    }

    pub fn field(&self) -> &str {
        &self.field
    }

    pub fn index(&self) -> &LexicalIndex {
        &self.index
    }

    /// (Re-)index a point from its payload. A point whose field is missing or
    /// not text is removed from the index (Qdrant likewise skips values whose
    /// type does not match a full-text index).
    pub fn upsert_point(&mut self, point: &Point) {
        match extract_text(&point.payload, &self.field) {
            Some(text) => self.index.index_document(point.id, &text),
            None => {
                self.index.remove_document(point.id);
            }
        }
    }

    pub fn remove_point(&mut self, id: PointId) {
        self.index.remove_document(id);
    }
}

/// Read `field` from a raw payload: a JSON string, or the strings of a JSON
/// array joined by spaces. Anything else is not text.
pub fn extract_text(payload: &HashMap<String, Vec<u8>>, field: &str) -> Option<String> {
    let bytes = payload.get(field)?;
    // Fast path: the field is almost always a plain JSON string.
    if let Ok(text) = serde_json::from_slice::<String>(bytes) {
        return Some(text);
    }
    match serde_json::from_slice::<serde_json::Value>(bytes).ok()? {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Array(items) => {
            let parts: Vec<&str> = items.iter().filter_map(|v| v.as_str()).collect();
            (!parts.is_empty()).then(|| parts.join(" "))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn point(id: PointId, field: &str, json: &str) -> Point {
        Point::new_vector(id, vec![0.0]).with_field(field, json.as_bytes().to_vec())
    }

    #[test]
    fn extracts_strings_and_string_arrays_only() {
        let p = point(1, "text", r#""hello world""#);
        assert_eq!(
            extract_text(&p.payload, "text").as_deref(),
            Some("hello world")
        );
        let p = point(1, "text", r#"["a", 1, "b"]"#);
        assert_eq!(extract_text(&p.payload, "text").as_deref(), Some("a b"));
        assert_eq!(extract_text(&point(1, "text", "42").payload, "text"), None);
        assert_eq!(extract_text(&p.payload, "missing"), None);
    }

    #[test]
    fn upsert_replaces_and_non_text_removes() {
        let mut idx = PayloadTextIndex::new(&TextIndexConfig::bm25("text"));
        idx.upsert_point(&point(1, "text", r#""red apple""#));
        assert_eq!(idx.index().search("apple", 10).len(), 1);
        idx.upsert_point(&point(1, "text", r#""green pear""#));
        assert!(idx.index().search("apple", 10).is_empty());
        assert_eq!(idx.index().search("pear", 10)[0].id, 1);
        idx.upsert_point(&point(1, "text", "7"));
        assert_eq!(idx.index().doc_count(), 0);
        idx.upsert_point(&point(2, "text", r#""pear""#));
        idx.remove_point(2);
        assert!(idx.index().search("pear", 10).is_empty());
    }
}
