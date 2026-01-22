//! Neo4j benchmark runner

use crate::datasets::{PersonData, RelationshipData};
use crate::Timer;
use neo4rs::{query, Graph};
use std::time::Duration;

/// Neo4j benchmark runner
pub struct Neo4jRunner {
    graph: Graph,
}

impl Neo4jRunner {
    /// Connect to a Neo4j instance
    pub async fn connect(
        uri: &str,
        user: &str,
        password: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let graph = Graph::new(uri, user, password).await?;
        Ok(Self { graph })
    }

    /// Clear all data from the database
    pub async fn clear(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Delete all relationships first, then all nodes
        self.graph.run(query("MATCH ()-[r]->() DELETE r")).await?;
        self.graph.run(query("MATCH (n) DELETE n")).await?;
        Ok(())
    }

    /// Execute a Cypher query
    pub async fn query(
        &self,
        cypher: &str,
    ) -> Result<Vec<neo4rs::Row>, Box<dyn std::error::Error + Send + Sync>> {
        let mut result = self.graph.execute(query(cypher)).await?;
        let mut rows = Vec::new();
        while let Some(row) = result.next().await? {
            rows.push(row);
        }
        Ok(rows)
    }

    /// Load Person data into Neo4j
    pub async fn load_people(
        &self,
        people: &[PersonData],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for person in people {
            let q = query("CREATE (p:Person {name: $name, age: $age, city: $city, email: $email})")
                .param("name", person.name.clone())
                .param("age", person.age)
                .param("city", person.city.clone())
                .param("email", person.email.clone());

            self.graph.run(q).await?;
        }
        Ok(())
    }

    /// Load relationships into Neo4j
    pub async fn load_relationships(
        &self,
        relationships: &[RelationshipData],
        people: &[PersonData],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // First ensure people have IDs we can reference
        for rel in relationships {
            if rel.from_idx >= people.len() || rel.to_idx >= people.len() {
                continue;
            }

            let from_name = &people[rel.from_idx].name;
            let to_name = &people[rel.to_idx].name;

            let q = query(&format!(
                r#"MATCH (a:Person {{name: $from_name}}), (b:Person {{name: $to_name}})
                   CREATE (a)-[:{}  {{weight: $weight}}]->(b)"#,
                rel.rel_type
            ))
            .param("from_name", from_name.clone())
            .param("to_name", to_name.clone())
            .param("weight", rel.weight);

            self.graph.run(q).await?;
        }
        Ok(())
    }

    /// Get node count
    pub async fn node_count(&self) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut result = self
            .graph
            .execute(query("MATCH (n) RETURN count(n) as count"))
            .await?;

        if let Some(row) = result.next().await? {
            let count: i64 = row.get("count")?;
            return Ok(count as usize);
        }
        Ok(0)
    }

    // === Benchmark methods ===

    /// Benchmark single node creation
    pub async fn bench_create_single(
        &self,
        person: &PersonData,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let q = query("CREATE (p:Person {name: $name, age: $age, city: $city, email: $email})")
            .param("name", person.name.clone())
            .param("age", person.age)
            .param("city", person.city.clone())
            .param("email", person.email.clone());

        let timer = Timer::start();
        self.graph.run(q).await?;
        Ok(timer.elapsed())
    }

    /// Benchmark match all nodes
    pub async fn bench_match_all(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let timer = Timer::start();
        self.query("MATCH (n) RETURN n").await?;
        Ok(timer.elapsed())
    }

    /// Benchmark match by label
    pub async fn bench_match_by_label(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let timer = Timer::start();
        self.query("MATCH (p:Person) RETURN p").await?;
        Ok(timer.elapsed())
    }

    /// Benchmark match by label with limit
    pub async fn bench_match_by_label_limit(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let timer = Timer::start();
        self.query("MATCH (p:Person) RETURN p LIMIT 10").await?;
        Ok(timer.elapsed())
    }

    /// Benchmark match with property filter
    pub async fn bench_match_with_filter(
        &self,
        min_age: i64,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let mut result = self
            .graph
            .execute(
                query("MATCH (p:Person) WHERE p.age > $min_age RETURN p").param("min_age", min_age),
            )
            .await?;

        let timer = Timer::start();
        while let Some(_row) = result.next().await? {}
        Ok(timer.elapsed())
    }

    /// Benchmark match with complex filter
    pub async fn bench_match_complex_filter(
        &self,
        min_age: i64,
        city: &str,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let timer = Timer::start();
        let mut result = self
            .graph
            .execute(
                query("MATCH (p:Person) WHERE p.age > $min_age AND p.city = $city RETURN p")
                    .param("min_age", min_age)
                    .param("city", city),
            )
            .await?;
        while let Some(_row) = result.next().await? {}
        Ok(timer.elapsed())
    }

    /// Benchmark property projection
    pub async fn bench_projection(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let timer = Timer::start();
        self.query("MATCH (p:Person) RETURN p.name, p.age, p.city LIMIT 100")
            .await?;
        Ok(timer.elapsed())
    }

    /// Benchmark ordering
    pub async fn bench_order_by(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let timer = Timer::start();
        self.query("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC LIMIT 100")
            .await?;
        Ok(timer.elapsed())
    }

    /// Benchmark skip/limit pagination
    pub async fn bench_skip_limit(
        &self,
    ) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let timer = Timer::start();
        self.query("MATCH (p:Person) RETURN p.name SKIP 50 LIMIT 50")
            .await?;
        Ok(timer.elapsed())
    }
}
