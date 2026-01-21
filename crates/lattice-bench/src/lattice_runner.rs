//! LatticeDB benchmark runner

use crate::datasets::{PersonData, RelationshipData};
use crate::Timer;
use lattice_core::cypher::{CypherHandler, DefaultCypherHandler};
use lattice_core::engine::collection::CollectionEngine;
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::value::CypherValue;
use std::collections::HashMap;
use std::time::Duration;

/// LatticeDB benchmark runner
pub struct LatticeRunner {
    engine: CollectionEngine,
    handler: DefaultCypherHandler,
}

impl LatticeRunner {
    /// Create a new LatticeDB runner with a fresh collection
    pub fn new(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = CollectionConfig::new(
            name,
            // Minimal vector config (graph-focused benchmark)
            VectorConfig::new(4, Distance::Cosine),
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 200,
            },
        );

        let engine = CollectionEngine::new(config)?;
        let handler = DefaultCypherHandler::new();

        Ok(Self { engine, handler })
    }

    /// Clear all data from the collection
    pub fn clear(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Create a fresh collection
        let config = self.engine.config().clone();
        self.engine = CollectionEngine::new(config)?;
        Ok(())
    }

    /// Execute a Cypher query
    pub fn query(
        &mut self,
        cypher: &str,
        params: HashMap<String, CypherValue>,
    ) -> Result<lattice_core::cypher::executor::QueryResult, Box<dyn std::error::Error>> {
        Ok(self.handler.query(cypher, &mut self.engine, params)?)
    }

    /// Load Person data into the collection
    pub fn load_people(&mut self, people: &[PersonData]) -> Result<(), Box<dyn std::error::Error>> {
        for person in people {
            let query = format!(
                r#"CREATE (p:Person {{name: "{}", age: {}, city: "{}", email: "{}"}})"#,
                person.name, person.age, person.city, person.email
            );
            self.query(&query, HashMap::new())?;
        }
        Ok(())
    }

    /// Load relationships into the collection
    pub fn load_relationships(
        &mut self,
        _relationships: &[RelationshipData],
        _node_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Relationships would require node IDs - skip for now
        // This benchmark focuses on node operations
        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.engine.point_count()
    }

    // === Benchmark methods ===

    /// Benchmark single node creation
    pub fn bench_create_single(
        &mut self,
        person: &PersonData,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let query = format!(
            r#"CREATE (p:Person {{name: "{}", age: {}, city: "{}", email: "{}"}})"#,
            person.name, person.age, person.city, person.email
        );

        let timer = Timer::start();
        self.query(&query, HashMap::new())?;
        Ok(timer.elapsed())
    }

    /// Benchmark match all nodes
    pub fn bench_match_all(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let timer = Timer::start();
        self.query("MATCH (n) RETURN n", HashMap::new())?;
        Ok(timer.elapsed())
    }

    /// Benchmark match by label
    pub fn bench_match_by_label(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let timer = Timer::start();
        self.query("MATCH (p:Person) RETURN p", HashMap::new())?;
        Ok(timer.elapsed())
    }

    /// Benchmark match by label with limit
    pub fn bench_match_by_label_limit(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let timer = Timer::start();
        self.query("MATCH (p:Person) RETURN p LIMIT 10", HashMap::new())?;
        Ok(timer.elapsed())
    }

    /// Benchmark match with property filter
    pub fn bench_match_with_filter(
        &mut self,
        min_age: i64,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let query = format!("MATCH (p:Person) WHERE p.age > {} RETURN p", min_age);

        let timer = Timer::start();
        self.query(&query, HashMap::new())?;
        Ok(timer.elapsed())
    }

    /// Benchmark match with complex filter
    pub fn bench_match_complex_filter(
        &mut self,
        min_age: i64,
        city: &str,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let query = format!(
            r#"MATCH (p:Person) WHERE p.age > {} AND p.city = "{}" RETURN p"#,
            min_age, city
        );

        let timer = Timer::start();
        self.query(&query, HashMap::new())?;
        Ok(timer.elapsed())
    }

    /// Benchmark property projection
    pub fn bench_projection(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let timer = Timer::start();
        self.query(
            "MATCH (p:Person) RETURN p.name, p.age, p.city LIMIT 100",
            HashMap::new(),
        )?;
        Ok(timer.elapsed())
    }

    /// Benchmark ordering
    pub fn bench_order_by(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let timer = Timer::start();
        self.query(
            "MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC LIMIT 100",
            HashMap::new(),
        )?;
        Ok(timer.elapsed())
    }

    /// Benchmark skip/limit pagination
    pub fn bench_skip_limit(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let timer = Timer::start();
        self.query(
            "MATCH (p:Person) RETURN p.name SKIP 50 LIMIT 50",
            HashMap::new(),
        )?;
        Ok(timer.elapsed())
    }
}
