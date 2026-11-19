//! Qdrant Query API DTOs for `POST /collections/{name}/points/query`.
//!
//! Accepted `query` forms (Qdrant shapes):
//! - dense vector: `[0.1, 0.2]`, `{"nearest": [0.1, 0.2]}`, `{"nearest": {"vector": [...]}}`
//! - text (inference "Document"): `{"text": "...", "model": "bm25"}` or wrapped
//!   in `{"nearest": {...}}`; `model` is `bm25` / `qdrant/bm25` for the lexical
//!   index, or the collection's embedding model for a dense search
//! - fusion of `prefetch` results: `{"fusion": "rrf"}`, `{"rrf": {}}`, `{"rrf": {"k": 60}}`

use super::request::{default_limit, default_true, SearchParams};
use serde::{Deserialize, Deserializer};

/// Top-level query request.
#[derive(Debug, Clone, Deserialize)]
pub struct QueryRequest {
    /// Sub-queries whose results a fusion `query` combines (one object or a list).
    #[serde(default, deserialize_with = "one_or_many")]
    pub prefetch: Vec<Prefetch>,
    pub query: QueryInput,
    /// Accepted for Qdrant-client compatibility and ignored: a collection holds
    /// one dense vector plus (optionally) one text index, and the query form
    /// already selects between them.
    #[serde(default)]
    pub using: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_true")]
    pub with_payload: bool,
    #[serde(default)]
    pub with_vector: bool,
    #[serde(default)]
    pub params: Option<SearchParams>,
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

/// A prefetch stage: a query of its own, optionally fusing nested prefetches.
#[derive(Debug, Clone, Deserialize)]
pub struct Prefetch {
    #[serde(default, deserialize_with = "one_or_many")]
    pub prefetch: Vec<Prefetch>,
    pub query: QueryInput,
    #[serde(default)]
    pub using: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub params: Option<SearchParams>,
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

/// Fusion methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusion {
    /// Reciprocal Rank Fusion with constant `k`.
    Rrf { k: u32 },
}

/// One parsed `query` value.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryInput {
    Vector(Vec<f32>),
    Text { text: String, model: String },
    Fusion(Fusion),
}

impl<'de> Deserialize<'de> for QueryInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        parse_query(&value).map_err(serde::de::Error::custom)
    }
}

fn one_or_many<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Prefetch>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        Many(Vec<Prefetch>),
        One(Box<Prefetch>),
    }
    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::Many(v) => v,
        OneOrMany::One(p) => vec![*p],
    })
}

const SUPPORTED: &str = "expected a vector, {\"nearest\": ...}, {\"text\": ..., \"model\": ...}, {\"fusion\": \"rrf\"} or {\"rrf\": {\"k\": n}}";

fn parse_query(value: &serde_json::Value) -> Result<QueryInput, String> {
    use serde_json::Value;
    let obj = match value {
        Value::Array(_) => return parse_vector(value).map(QueryInput::Vector),
        Value::Object(obj) => obj,
        _ => return Err(SUPPORTED.to_string()),
    };
    if let Some(inner) = obj.get("nearest") {
        return match inner {
            Value::Object(o) if o.contains_key("text") => parse_document(o),
            _ => parse_vector(inner).map(QueryInput::Vector),
        };
    }
    if obj.contains_key("text") {
        return parse_document(obj);
    }
    if let Some(fusion) = obj.get("fusion") {
        return match fusion.as_str() {
            Some("rrf") => Ok(QueryInput::Fusion(Fusion::Rrf {
                k: lattice_core::text::RRF_K,
            })),
            Some("dbsf") => Err("fusion 'dbsf' is not supported; use 'rrf'".to_string()),
            _ => Err(format!("unknown fusion {fusion}; use 'rrf'")),
        };
    }
    if let Some(rrf) = obj.get("rrf") {
        let k = match rrf.get("k") {
            None => lattice_core::text::RRF_K,
            Some(k) => k
                .as_u64()
                .and_then(|k| u32::try_from(k).ok())
                .filter(|&k| k >= 1)
                .ok_or_else(|| format!("rrf.k must be a positive integer, got {k}"))?,
        };
        return Ok(QueryInput::Fusion(Fusion::Rrf { k }));
    }
    Err(SUPPORTED.to_string())
}

fn parse_document(obj: &serde_json::Map<String, serde_json::Value>) -> Result<QueryInput, String> {
    let text = obj
        .get("text")
        .and_then(|t| t.as_str())
        .ok_or("'text' must be a string")?;
    // No implicit model: guessing one would silently pick a vector space.
    let model = obj
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or("text queries require 'model': \"bm25\" or the collection's embedding model")?;
    Ok(QueryInput::Text {
        text: text.to_string(),
        model: model.to_string(),
    })
}

/// Parse a dense vector, rejecting values that are not finite f32s.
fn parse_vector(value: &serde_json::Value) -> Result<Vec<f32>, String> {
    match value {
        serde_json::Value::Array(arr) => arr
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_f64()
                    .filter(|f| f.is_finite() && f.abs() <= f64::from(f32::MAX))
                    .map(|f| f as f32)
                    .ok_or_else(|| {
                        format!("vector element {i} must be a finite number within f32 range")
                    })
            })
            .collect(),
        serde_json::Value::Object(obj) => match obj.get("vector") {
            Some(inner) => parse_vector(inner),
            None => Err("expected 'vector' field in query object".to_string()),
        },
        _ => Err("expected array or object for vector".to_string()),
    }
}

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;
