//! Collection and HNSW configuration types
//!
//! # PCND Compliance
//!
//! All HNSW parameters are REQUIRED - no implicit defaults.
//! This prevents production surprises from unexpected default values.

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Distance metric for vector similarity
///
/// Determines how similarity between vectors is calculated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distance {
    /// Cosine similarity (angle between vectors)
    ///
    /// Range: [0, 2] where 0 = identical, 2 = opposite
    /// Best for: Normalized embeddings, text similarity
    Cosine,

    /// Euclidean distance (L2 norm)
    ///
    /// Range: [0, ∞) where 0 = identical
    /// Best for: Spatial data, image embeddings
    Euclid,

    /// Dot product (inner product)
    ///
    /// Range: (-∞, ∞) where higher = more similar
    /// Best for: Maximum inner product search (MIPS)
    Dot,
}

/// HNSW index configuration
///
/// # PCND Compliance
///
/// ALL parameters are REQUIRED. There is no `Default` implementation.
/// This is intentional - HNSW performance is highly sensitive to these values,
/// and implicit defaults lead to unexpected behavior in production.
///
/// # Parameter Guide
///
/// | Parameter | Typical Range | Effect |
/// |-----------|--------------|--------|
/// | `m` | 5-48 | Higher = better recall, more memory |
/// | `m0` | 2*m | Layer 0 has more connections |
/// | `ml` | 1/ln(m) | Level distribution |
/// | `ef` | 10-500 | Search queue size (quality vs speed) |
/// | `ef_construction` | 100-800 | Build quality (one-time cost) |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum connections per node in layers 1+ (upper layers)
    ///
    /// Typical range: 5-48
    /// - Higher values improve recall but increase memory usage
    /// - 16 is a common starting point
    pub m: usize,

    /// Maximum connections per node in layer 0 (ground layer)
    ///
    /// Paper recommendation: m0 = 2 * m
    /// Layer 0 benefits from more connections since most searches end there.
    pub m0: usize,

    /// Level distribution multiplier
    ///
    /// Formula: 1 / ln(m)
    /// Controls probability of assigning nodes to higher layers.
    /// Higher ml = more nodes in upper layers = faster search but more memory.
    pub ml: f64,

    /// Search queue size during queries
    ///
    /// Typical range: 10-500
    /// - Higher = better recall, slower search
    /// - Must be >= k (number of results requested)
    /// - Can be overridden per-query
    pub ef: usize,

    /// Search queue size during index construction
    ///
    /// Typical range: 100-800
    /// - Higher = better graph quality, slower insertion
    /// - One-time cost at build time
    /// - 200-500 is a good starting point
    pub ef_construction: usize,
}

impl HnswConfig {
    /// Validate configuration parameters
    ///
    /// Returns error if any parameter is invalid.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.m < 2 {
            return Err(ConfigError::InvalidParameter {
                name: "m",
                message: "must be >= 2".into(),
            });
        }

        if self.m0 < self.m {
            return Err(ConfigError::InvalidParameter {
                name: "m0",
                message: format!("must be >= m ({})", self.m),
            });
        }

        if self.ml <= 0.0 {
            return Err(ConfigError::InvalidParameter {
                name: "ml",
                message: "must be > 0".into(),
            });
        }

        if self.ef < 1 {
            return Err(ConfigError::InvalidParameter {
                name: "ef",
                message: "must be >= 1".into(),
            });
        }

        if self.ef_construction < 1 {
            return Err(ConfigError::InvalidParameter {
                name: "ef_construction",
                message: "must be >= 1".into(),
            });
        }

        Ok(())
    }

    /// Calculate the recommended ml value for a given m
    ///
    /// Formula: 1 / ln(m)
    pub fn recommended_ml(m: usize) -> f64 {
        1.0 / (m as f64).ln()
    }
}

/// Vector configuration for a collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    /// Dimensionality of vectors
    ///
    /// All vectors in the collection must have this exact dimension.
    pub size: usize,

    /// Distance metric for similarity calculation
    pub distance: Distance,
}

impl VectorConfig {
    /// Create a new vector configuration
    pub fn new(size: usize, distance: Distance) -> Self {
        Self { size, distance }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.size == 0 {
            return Err(ConfigError::InvalidParameter {
                name: "size",
                message: "must be > 0".into(),
            });
        }
        Ok(())
    }
}

/// Collection configuration - SSOT for all collection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Collection name (unique identifier)
    pub name: String,

    /// Vector configuration
    pub vectors: VectorConfig,

    /// HNSW index configuration
    pub hnsw: HnswConfig,

    /// Relation name -> relation_id mapping for graph edges
    ///
    /// Example: {"similar_to": 0, "related_to": 1}
    /// The u16 IDs are used in Edge.relation_id for compact storage.
    #[serde(default)]
    pub relations: HashMap<String, u16>,
}

impl CollectionConfig {
    /// Create a new collection configuration
    pub fn new(name: impl Into<String>, vectors: VectorConfig, hnsw: HnswConfig) -> Self {
        Self {
            name: name.into(),
            vectors,
            hnsw,
            relations: HashMap::new(),
        }
    }

    /// Add a relation type
    pub fn with_relation(mut self, name: impl Into<String>, id: u16) -> Self {
        self.relations.insert(name.into(), id);
        self
    }

    /// Validate the entire configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.name.is_empty() {
            return Err(ConfigError::InvalidParameter {
                name: "name",
                message: "must not be empty".into(),
            });
        }

        self.vectors.validate()?;
        self.hnsw.validate()?;

        Ok(())
    }

    /// Get relation ID by name
    pub fn relation_id(&self, name: &str) -> Option<u16> {
        self.relations.get(name).copied()
    }

    /// Get relation name by ID
    pub fn relation_name(&self, id: u16) -> Option<&str> {
        self.relations
            .iter()
            .find(|(_, &v)| v == id)
            .map(|(k, _)| k.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_hnsw_config_validation() {
        // Valid config
        let config = HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        };
        assert!(config.validate().is_ok());

        // Invalid m
        let config = HnswConfig { m: 1, ..config.clone() };
        assert!(config.validate().is_err());

        // Invalid m0 < m
        let config = HnswConfig { m0: 8, ..HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        }};
        assert!(config.validate().is_err());

        // Invalid ml
        let config = HnswConfig { ml: 0.0, ..HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        }};
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_recommended_ml() {
        // For m=16, ml ≈ 0.36
        let ml = HnswConfig::recommended_ml(16);
        assert!((ml - 0.36).abs() < 0.01);

        // For m=32, ml ≈ 0.29
        let ml = HnswConfig::recommended_ml(32);
        assert!((ml - 0.29).abs() < 0.01);
    }

    #[test]
    fn test_collection_config() {
        let config = CollectionConfig::new(
            "test_collection",
            VectorConfig::new(128, Distance::Cosine),
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 200,
            },
        )
        .with_relation("similar_to", 0)
        .with_relation("related_to", 1);

        assert!(config.validate().is_ok());
        assert_eq!(config.relation_id("similar_to"), Some(0));
        assert_eq!(config.relation_name(1), Some("related_to"));
    }

    #[test]
    fn test_collection_config_empty_name() {
        let config = CollectionConfig::new(
            "",
            VectorConfig::new(128, Distance::Cosine),
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 200,
            },
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vector_config_zero_size() {
        let config = VectorConfig::new(0, Distance::Cosine);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_distance_serde() {
        // Test JSON serialization matches Qdrant format
        assert_eq!(
            serde_json::to_string(&Distance::Cosine).unwrap(),
            r#""cosine""#
        );
        assert_eq!(
            serde_json::to_string(&Distance::Euclid).unwrap(),
            r#""euclid""#
        );
        assert_eq!(
            serde_json::to_string(&Distance::Dot).unwrap(),
            r#""dot""#
        );
    }
}
