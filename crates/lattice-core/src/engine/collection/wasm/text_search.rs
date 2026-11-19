//! BM25 search for the wasm engine.

use super::CollectionEngine;
use crate::error::LatticeResult;
use crate::text::{no_text_index_error, PayloadTextIndex, TextSearchQuery};
use crate::types::query::SearchResult;

impl CollectionEngine {
    /// The collection's BM25 index, if it configures one.
    pub fn text_index(&self) -> Option<&PayloadTextIndex> {
        self.text_index.as_ref()
    }

    /// Rank points by BM25 score (higher is better) against the text index.
    pub fn search_text(&self, query: &TextSearchQuery) -> LatticeResult<Vec<SearchResult>> {
        let text_index = self
            .text_index
            .as_ref()
            .ok_or_else(|| no_text_index_error(&self.config.name))?;
        let hits = text_index.index().search(&query.text, query.limit);
        Ok(hits
            .into_iter()
            .filter_map(|hit| {
                let point = self.points.get(&hit.id)?;
                let mut result = SearchResult::new(hit.id, hit.score);
                if query.with_vector {
                    result = result.with_vector(point.vector.clone());
                }
                if query.with_payload {
                    result = result.with_payload(self.payload_to_json(&point.payload));
                }
                Some(result)
            })
            .collect())
    }
}
