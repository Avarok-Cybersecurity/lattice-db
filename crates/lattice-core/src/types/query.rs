//! Query and result types for search operations

use crate::types::point::{PointId, Vector};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Search query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query vector
    pub vector: Vector,

    /// Number of results to return
    pub limit: usize,

    /// Search queue size (overrides collection default)
    ///
    /// Higher values improve recall at the cost of speed.
    /// Must be >= limit.
    #[serde(default)]
    pub ef: Option<usize>,

    /// Include vector data in results
    #[serde(default)]
    pub with_vector: bool,

    /// Include payload data in results
    #[serde(default)]
    pub with_payload: bool,

    /// Score threshold (filter results below this score)
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

impl SearchQuery {
    /// Create a new search query
    pub fn new(vector: Vector, limit: usize) -> Self {
        Self {
            vector,
            limit,
            ef: None,
            with_vector: false,
            with_payload: true,
            score_threshold: None,
        }
    }

    /// Set ef parameter
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    /// Include vector data in results
    pub fn include_vector(mut self) -> Self {
        self.with_vector = true;
        self
    }

    /// Include payload data in results
    pub fn include_payload(mut self) -> Self {
        self.with_payload = true;
        self
    }

    /// Set score threshold
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
}

/// Single search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Point ID
    pub id: PointId,

    /// Similarity score (interpretation depends on distance metric)
    ///
    /// - Cosine: [0, 2] where 0 = identical
    /// - Euclid: [0, ∞) where 0 = identical
    /// - Dot: (-∞, ∞) where higher = more similar
    pub score: f32,

    /// Vector data (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vector>,

    /// Payload data (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(id: PointId, score: f32) -> Self {
        Self {
            id,
            score,
            vector: None,
            payload: None,
        }
    }

    /// Add vector data
    pub fn with_vector(mut self, vector: Vector) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Add payload data
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && (self.score - other.score).abs() < f32::EPSILON
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort by score ascending (lower distance = better match)
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Scroll query for paginated point retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollQuery {
    /// Number of points to return
    pub limit: usize,

    /// Offset for pagination (point ID to start after)
    #[serde(default)]
    pub offset: Option<PointId>,

    /// Include vector data
    #[serde(default)]
    pub with_vector: bool,

    /// Include payload data
    #[serde(default = "default_true")]
    pub with_payload: bool,
}

fn default_true() -> bool {
    true
}

impl ScrollQuery {
    /// Create a new scroll query
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            offset: None,
            with_vector: false,
            with_payload: true,
        }
    }

    /// Set offset
    pub fn with_offset(mut self, offset: PointId) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Include vector data in results
    pub fn include_vector(mut self) -> Self {
        self.with_vector = true;
        self
    }

    /// Include payload data in results
    pub fn include_payload(mut self) -> Self {
        self.with_payload = true;
        self
    }
}

/// Scroll result with pagination info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollResult {
    /// Points in this page
    pub points: Vec<ScrollPoint>,

    /// Next offset (None if no more results)
    pub next_offset: Option<PointId>,
}

/// Point data for scroll results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollPoint {
    /// Point ID
    pub id: PointId,

    /// Vector data (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vector>,

    /// Payload data (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_search_query_builder() {
        let query = SearchQuery::new(vec![0.1, 0.2, 0.3], 10)
            .with_ef(100)
            .include_vector()
            .with_score_threshold(0.5);

        assert_eq!(query.limit, 10);
        assert_eq!(query.ef, Some(100));
        assert!(query.with_vector);
        assert_eq!(query.score_threshold, Some(0.5));
    }

    #[test]
    fn test_search_result_ordering() {
        let mut results = vec![
            SearchResult::new(1, 0.5),
            SearchResult::new(2, 0.1),
            SearchResult::new(3, 0.9),
        ];

        results.sort();

        // Should be sorted by score ascending (lower = better)
        assert_eq!(results[0].id, 2); // 0.1
        assert_eq!(results[1].id, 1); // 0.5
        assert_eq!(results[2].id, 3); // 0.9
    }

    #[test]
    fn test_scroll_query() {
        let query = ScrollQuery::new(100).with_offset(42);

        assert_eq!(query.limit, 100);
        assert_eq!(query.offset, Some(42));
        assert!(query.with_payload);
        assert!(!query.with_vector);
    }
}
