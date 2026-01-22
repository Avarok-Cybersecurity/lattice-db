//! HTTP-based LatticeDB benchmark runner
//!
//! This runner connects to a running LatticeDB server via HTTP,
//! providing fair comparison with Neo4j (both include network latency).

use crate::datasets::PersonData;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// HTTP request body for Cypher queries
#[derive(Serialize)]
struct CypherRequest {
    query: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    parameters: HashMap<String, serde_json::Value>,
}

/// HTTP response from Cypher queries
#[derive(Deserialize)]
struct ApiResponse<T> {
    result: Option<T>,
    status: String,
}

#[derive(Deserialize)]
pub struct CypherResponse {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub stats: CypherStats,
}

#[derive(Deserialize)]
pub struct CypherStats {
    pub nodes_created: u64,
    pub relationships_created: u64,
    pub nodes_deleted: u64,
    pub relationships_deleted: u64,
    pub properties_set: u64,
    pub execution_time_ms: u64,
}

/// Collection creation request
#[derive(Serialize)]
struct CreateCollectionRequest {
    vectors: VectorParams,
}

#[derive(Serialize)]
struct VectorParams {
    size: usize,
    distance: String,
}

/// HTTP-based LatticeDB benchmark runner
pub struct HttpRunner {
    client: Client,
    base_url: String,
    collection: String,
}

impl HttpRunner {
    /// Create a new HTTP runner and set up the collection
    pub async fn new(
        base_url: &str,
        collection: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = Client::new();
        let runner = Self {
            client,
            base_url: base_url.to_string(),
            collection: collection.to_string(),
        };

        // Create collection if it doesn't exist
        runner.create_collection().await?;

        Ok(runner)
    }

    /// Check if the server is available
    pub async fn is_available(&self) -> bool {
        let url = format!("{}/collections", self.base_url);
        self.client.get(&url).send().await.is_ok()
    }

    /// Create the collection
    async fn create_collection(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);

        // Delete existing collection
        let _ = self.client.delete(&url).send().await;

        // Create new collection
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 4,
                distance: "Cosine".to_string(),
            },
        };

        self.client.put(&url).json(&request).send().await?;
        Ok(())
    }

    /// Clear all data from the collection
    pub async fn clear(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.create_collection().await
    }

    /// Execute a Cypher query
    pub async fn query(
        &self,
        cypher: &str,
    ) -> Result<CypherResponse, Box<dyn std::error::Error + Send + Sync>> {
        self.query_with_params(cypher, HashMap::new()).await
    }

    /// Execute a Cypher query with parameters
    pub async fn query_with_params(
        &self,
        cypher: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<CypherResponse, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!(
            "{}/collections/{}/graph/query",
            self.base_url, self.collection
        );

        let request = CypherRequest {
            query: cypher.to_string(),
            parameters: params,
        };

        let response = self.client.post(&url).json(&request).send().await?;

        let api_response: ApiResponse<CypherResponse> = response.json().await?;

        api_response
            .result
            .ok_or_else(|| "Query failed: no result".into())
    }

    /// Load Person data into the collection
    pub async fn load_people(
        &self,
        people: &[PersonData],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for person in people {
            let query = format!(
                r#"CREATE (p:Person {{name: "{}", age: {}, city: "{}", email: "{}"}})"#,
                person.name, person.age, person.city, person.email
            );
            self.query(&query).await?;
        }
        Ok(())
    }

    // === Benchmark methods ===

    /// Benchmark single node creation
    pub async fn bench_create_single(
        &self,
        person: &PersonData,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let query = format!(
            r#"CREATE (p:Person {{name: "{}", age: {}, city: "{}", email: "{}"}})"#,
            person.name, person.age, person.city, person.email
        );

        let start = Instant::now();
        self.query(&query).await?;
        Ok(start.elapsed())
    }

    /// Benchmark match all nodes
    pub async fn bench_match_all(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.query("MATCH (n) RETURN n").await?;
        Ok(start.elapsed())
    }

    /// Benchmark match by label
    pub async fn bench_match_by_label(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.query("MATCH (p:Person) RETURN p").await?;
        Ok(start.elapsed())
    }

    /// Benchmark match by label with limit
    pub async fn bench_match_by_label_limit(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.query("MATCH (p:Person) RETURN p LIMIT 10").await?;
        Ok(start.elapsed())
    }

    /// Benchmark match with property filter
    pub async fn bench_match_with_filter(
        &self,
        min_age: i64,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let query = format!("MATCH (p:Person) WHERE p.age > {} RETURN p", min_age);

        let start = Instant::now();
        self.query(&query).await?;
        Ok(start.elapsed())
    }

    /// Benchmark match with complex filter
    pub async fn bench_match_complex_filter(
        &self,
        min_age: i64,
        city: &str,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let query = format!(
            r#"MATCH (p:Person) WHERE p.age > {} AND p.city = "{}" RETURN p"#,
            min_age, city
        );

        let start = Instant::now();
        self.query(&query).await?;
        Ok(start.elapsed())
    }

    /// Benchmark property projection
    pub async fn bench_projection(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.query("MATCH (p:Person) RETURN p.name, p.age, p.city LIMIT 100")
            .await?;
        Ok(start.elapsed())
    }

    /// Benchmark ordering
    pub async fn bench_order_by(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.query("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC LIMIT 100")
            .await?;
        Ok(start.elapsed())
    }

    /// Benchmark skip/limit pagination
    pub async fn bench_skip_limit(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.query("MATCH (p:Person) RETURN p.name SKIP 50 LIMIT 50")
            .await?;
        Ok(start.elapsed())
    }
}
