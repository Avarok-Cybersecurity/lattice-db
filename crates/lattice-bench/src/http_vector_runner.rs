//! HTTP-based LatticeDB vector benchmark runner
//!
//! Connects to a running LatticeDB server via HTTP for fair comparison with Qdrant
//! (both include network latency).
//!
//! ## Prerequisites
//!
//! Start LatticeDB server:
//! ```bash
//! cargo run --release -p lattice-server
//! ```

use crate::Timer;
use rand::prelude::*;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// LatticeDB HTTP vector benchmark runner
pub struct HttpVectorRunner {
    client: Client,
    base_url: String,
    collection_name: String,
    vector_dim: usize,
}

// === Request/Response DTOs ===

#[derive(Serialize)]
struct CreateCollectionRequest {
    vectors: VectorConfig,
}

#[derive(Serialize)]
struct VectorConfig {
    size: usize,
    distance: String,
}

#[derive(Serialize)]
struct UpsertRequest {
    points: Vec<PointStruct>,
}

#[derive(Serialize, Clone)]
struct PointStruct {
    id: u64,
    vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct SearchRequest {
    vector: Vec<f32>,
    limit: usize,
    with_payload: bool,
    with_vector: bool,
}

#[derive(Serialize)]
struct ScrollRequest {
    limit: usize,
    with_payload: bool,
    with_vector: bool,
}

#[derive(Serialize)]
struct GetPointsRequest {
    ids: Vec<u64>,
    with_payload: bool,
    with_vector: bool,
}

#[derive(Deserialize)]
struct ApiResponse<T> {
    #[allow(dead_code)]
    status: String,
    #[serde(default)]
    result: Option<T>,
}

impl HttpVectorRunner {
    /// Create a new HTTP vector runner
    pub fn new(
        base_url: &str,
        collection_name: &str,
        vector_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

        Ok(Self {
            client,
            base_url: base_url.to_string(),
            collection_name: collection_name.to_string(),
            vector_dim,
        })
    }

    /// Create collection with specified vector config
    pub fn create_collection(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Delete existing collection (ignore errors)
        let _ = self
            .client
            .delete(&format!(
                "{}/collections/{}",
                self.base_url, self.collection_name
            ))
            .send();

        // Create new collection
        let body = CreateCollectionRequest {
            vectors: VectorConfig {
                size: self.vector_dim,
                distance: "Cosine".to_string(),
            },
        };

        let resp = self
            .client
            .put(&format!(
                "{}/collections/{}",
                self.base_url, self.collection_name
            ))
            .json(&body)
            .send()?;

        if !resp.status().is_success() {
            return Err(format!("Failed to create collection: {}", resp.status()).into());
        }

        Ok(())
    }

    /// Generate random vector
    fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Load test data into LatticeDB via HTTP
    pub fn load_data(&self, count: usize, seed: u64) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let batch_size = 100;

        for batch_start in (0..count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(count);
            let points: Vec<PointStruct> = (batch_start..batch_end)
                .map(|i| {
                    let mut payload = HashMap::new();
                    payload.insert("index".to_string(), serde_json::json!(i));
                    payload.insert("name".to_string(), serde_json::json!(format!("point_{}", i)));
                    let categories = ["A", "B", "C"];
                    payload.insert("category".to_string(), serde_json::json!(categories[i % 3]));

                    PointStruct {
                        id: i as u64,
                        vector: Self::random_vector(self.vector_dim, &mut rng),
                        payload: Some(payload),
                    }
                })
                .collect();

            let body = UpsertRequest { points };

            let resp = self
                .client
                .put(&format!(
                    "{}/collections/{}/points",
                    self.base_url, self.collection_name
                ))
                .json(&body)
                .send()?;

            if !resp.status().is_success() {
                return Err(format!("Failed to upsert points: {}", resp.status()).into());
            }
        }

        // Small delay to allow indexing
        std::thread::sleep(Duration::from_millis(50));

        Ok(())
    }

    // === Benchmark methods ===

    /// Benchmark single point upsert
    pub fn bench_upsert(&self, seed: u64) -> Result<Duration, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let point = PointStruct {
            id: 999_999,
            vector: Self::random_vector(self.vector_dim, &mut rng),
            payload: Some({
                let mut p = HashMap::new();
                p.insert("bench".to_string(), serde_json::json!(true));
                p
            }),
        };

        let body = UpsertRequest {
            points: vec![point],
        };

        let timer = Timer::start();
        let resp = self
            .client
            .put(&format!(
                "{}/collections/{}/points",
                self.base_url, self.collection_name
            ))
            .json(&body)
            .send()?;

        let elapsed = timer.elapsed();

        if !resp.status().is_success() {
            return Err(format!("Upsert failed: {}", resp.status()).into());
        }

        Ok(elapsed)
    }

    /// Benchmark vector search
    pub fn bench_search(
        &self,
        k: usize,
        seed: u64,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let query_vector = Self::random_vector(self.vector_dim, &mut rng);

        let body = SearchRequest {
            vector: query_vector,
            limit: k,
            with_payload: false,
            with_vector: false,
        };

        let timer = Timer::start();
        let resp = self
            .client
            .post(&format!(
                "{}/collections/{}/points/search",
                self.base_url, self.collection_name
            ))
            .json(&body)
            .send()?;

        let elapsed = timer.elapsed();

        if !resp.status().is_success() {
            return Err(format!("Search failed: {}", resp.status()).into());
        }

        Ok(elapsed)
    }

    /// Benchmark point retrieval by ID
    pub fn bench_retrieve(&self, ids: &[u64]) -> Result<Duration, Box<dyn std::error::Error>> {
        let body = GetPointsRequest {
            ids: ids.to_vec(),
            with_payload: true,
            with_vector: true,
        };

        let timer = Timer::start();
        let resp = self
            .client
            .post(&format!(
                "{}/collections/{}/points",
                self.base_url, self.collection_name
            ))
            .json(&body)
            .send()?;

        let elapsed = timer.elapsed();

        if !resp.status().is_success() {
            return Err(format!("Retrieve failed: {}", resp.status()).into());
        }

        Ok(elapsed)
    }

    /// Benchmark scroll pagination
    pub fn bench_scroll(&self, limit: usize) -> Result<Duration, Box<dyn std::error::Error>> {
        let body = ScrollRequest {
            limit,
            with_payload: true,
            with_vector: false,
        };

        let timer = Timer::start();
        let resp = self
            .client
            .post(&format!(
                "{}/collections/{}/points/scroll",
                self.base_url, self.collection_name
            ))
            .json(&body)
            .send()?;

        let elapsed = timer.elapsed();

        if !resp.status().is_success() {
            return Err(format!("Scroll failed: {}", resp.status()).into());
        }

        Ok(elapsed)
    }

    /// Benchmark batch search (multiple sequential searches)
    pub fn bench_batch_search(
        &self,
        queries: usize,
        k: usize,
        seed: u64,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total = Duration::ZERO;

        for _ in 0..queries {
            let query_vector = Self::random_vector(self.vector_dim, &mut rng);
            let body = SearchRequest {
                vector: query_vector,
                limit: k,
                with_payload: false,
                with_vector: false,
            };

            let timer = Timer::start();
            let resp = self
                .client
                .post(&format!(
                    "{}/collections/{}/points/search",
                    self.base_url, self.collection_name
                ))
                .json(&body)
                .send()?;

            total += timer.elapsed();

            if !resp.status().is_success() {
                return Err(format!("Batch search failed: {}", resp.status()).into());
            }
        }

        Ok(total)
    }
}

/// Check if LatticeDB server is available
pub fn lattice_http_available() -> bool {
    let client = match Client::builder().timeout(Duration::from_secs(2)).build() {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Try port 6335 first (default for LatticeDB when Qdrant is on 6334)
    // then fall back to 6334 for standalone use
    client
        .get("http://localhost:6335/collections")
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or_else(|_| {
            client
                .get("http://localhost:6334/collections")
                .send()
                .map(|r| r.status().is_success())
                .unwrap_or(false)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires running LatticeDB server
    fn test_http_vector_runner() {
        if !lattice_http_available() {
            eprintln!("LatticeDB server not available, skipping test");
            return;
        }

        let runner =
            HttpVectorRunner::new("http://localhost:6334", "test_http_vector", 128).unwrap();

        runner.create_collection().unwrap();
        runner.load_data(100, 42).unwrap();

        let search_time = runner.bench_search(10, 42).unwrap();
        println!("HTTP Search time: {:?}", search_time);

        let retrieve_time = runner.bench_retrieve(&[0, 1, 2]).unwrap();
        println!("HTTP Retrieve time: {:?}", retrieve_time);
    }
}
