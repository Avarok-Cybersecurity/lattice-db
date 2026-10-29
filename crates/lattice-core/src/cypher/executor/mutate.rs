//! Mutation execution: CREATE and DELETE.

//! Query operation execution

use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats};
use crate::cypher::ast::{Expr, MapLiteral};
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::{ExecutorRow, Row};
use crate::types::point::{Point, PointId};
use crate::types::value::CypherValue;
use std::collections::HashMap;

impl QueryExecutor {
    /// Execute CreateNode operation
    pub(crate) fn execute_create_node(
        &self,
        labels: &Vec<String>,
        properties: &MapLiteral,
        _variable: Option<&String>,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Generate a new ID (hash of timestamp + random)
        let id = self.generate_point_id();

        // Build payload with labels and properties
        // Pre-allocate: 1 for _labels + number of properties
        let mut payload = HashMap::with_capacity(1 + properties.entries.len());

        // Compute label bitmap for O(1) label checks
        let label_bitmap = if !labels.is_empty() {
            // Store labels as JSON array (for backwards compatibility)
            let labels_json: Vec<&str> = labels.iter().map(|l| l.as_str()).collect();
            let labels_bytes =
                serde_json::to_vec(&labels_json).map_err(|e| CypherError::Internal {
                    message: e.to_string(),
                })?;
            payload.insert("_labels".to_string(), labels_bytes);

            // Compute bitmap for fast label checks
            ctx.labels_to_bitmap(labels)
        } else {
            0
        };

        // Store properties - use an empty slice as we're creating a new node
        let empty_row: ExecutorRow = ExecutorRow::new();
        for (key, value_expr) in &properties.entries {
            // Evaluate the expression to get the value
            let value = self.evaluate_expr(value_expr, &empty_row, ctx)?;
            let value_bytes = self.cypher_value_to_json_bytes(&value)?;
            payload.insert(key.to_string(), value_bytes);
        }

        // Create zero vector for graph-only node
        let dim = ctx.collection.vector_dim();
        let vector = vec![0.0f32; dim];

        let point = Point::new_vector(id, vector)
            .with_payload(payload)
            .with_label_bitmap(label_bitmap);

        ctx.collection
            .upsert_points(vec![point])
            .map_err(|e| CypherError::Internal {
                message: e.to_string(),
            })?;

        let mut stats = QueryStats::default();
        stats.nodes_created = 1;

        // Return the created node reference
        let row = ExecutorRow::single(CypherValue::NodeRef(id));

        Ok((vec![row], stats))
    }

    /// Execute CreateRelationship operation
    pub(crate) fn execute_create_relationship(
        &self,
        from: &String,
        to: &String,
        rel_type: &String,
        properties: &MapLiteral,
        _variable: Option<&String>,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // For now, we need the from/to IDs to be provided via parameters
        // This is a simplified implementation
        let from_id = ctx
            .parameters
            .get(from.as_str())
            .and_then(|v| match v {
                CypherValue::NodeRef(id) => Some(*id),
                // Only accept non-negative integers as valid point IDs
                CypherValue::Int(id) if *id >= 0 => Some(*id as u64),
                _ => None,
            })
            .ok_or_else(|| CypherError::unknown_variable(from.as_str()))?;

        let to_id = ctx
            .parameters
            .get(to.as_str())
            .and_then(|v| match v {
                CypherValue::NodeRef(id) => Some(*id),
                // Only accept non-negative integers as valid point IDs
                CypherValue::Int(id) if *id >= 0 => Some(*id as u64),
                _ => None,
            })
            .ok_or_else(|| CypherError::unknown_variable(to.as_str()))?;

        // Get weight from properties or default to 1.0
        let weight = properties
            .entries
            .iter()
            .find(|(k, _)| k.as_str() == "weight")
            .and_then(|(_, v)| {
                if let Expr::Literal(CypherValue::Float(f)) = &**v {
                    Some(*f as f32)
                } else if let Expr::Literal(CypherValue::Int(i)) = &**v {
                    Some(*i as f32)
                } else {
                    None
                }
            })
            .unwrap_or(1.0);

        ctx.collection
            .add_edge(from_id, to_id, rel_type.as_str(), weight)
            .map_err(|e| CypherError::Internal {
                message: e.to_string(),
            })?;

        let mut stats = QueryStats::default();
        stats.relationships_created = 1;

        // Return the created relationship as a reference
        let rel_id = self.generate_point_id(); // Generate unique ID for the relationship
        let row = ExecutorRow::single(CypherValue::RelationshipRef(rel_id));

        Ok((vec![row], stats))
    }

    /// Execute DeleteNode operation
    ///
    /// If `detach` is true, removes all outgoing edges from the node first.
    /// Note: Incoming edges (where this node is the target) are not removed
    /// as this would require scanning all nodes. Use explicit relationship
    /// deletion for complete cleanup.
    pub(crate) fn execute_delete_node(
        &self,
        input: &LogicalOp,
        _variable: &String,
        detach: bool,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, _) = self.execute_op(input, ctx)?;

        let mut deleted = 0u64;
        let mut relationships_deleted = 0u64;

        for row in input_rows {
            // Find the node ID in the row
            for value in row.iter() {
                if let CypherValue::NodeRef(id) = value {
                    // If detach, remove all outgoing relationships first
                    if detach {
                        if let Ok(edges) = ctx.collection.get_edges(*id) {
                            for edge in edges {
                                if ctx
                                    .collection
                                    .remove_edge(*id, edge.target_id, Some(&edge.relation))
                                    .is_ok()
                                {
                                    relationships_deleted += 1;
                                }
                            }
                        }
                    }

                    deleted += ctx.collection.delete_points(&[*id])? as u64;
                }
            }
        }

        let mut stats = QueryStats::default();
        stats.nodes_deleted = deleted;
        stats.relationships_deleted = relationships_deleted;

        Ok((Vec::new(), stats))
    }

    /// Generate a unique point ID
    pub(crate) fn generate_point_id(&self) -> PointId {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);

        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{Duration, SystemTime, UNIX_EPOCH};

            // Use timestamp with atomic counter fallback for robustness
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64;

            // Combine timestamp with counter to ensure uniqueness even if time fails
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut hash = timestamp ^ counter;
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xff51afd7ed558ccd);
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
            hash ^= hash >> 33;

            hash
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Use an incrementing counter with some entropy mixing
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut hash = counter;
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xff51afd7ed558ccd);
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
            hash ^= hash >> 33;

            hash
        }
    }
}
