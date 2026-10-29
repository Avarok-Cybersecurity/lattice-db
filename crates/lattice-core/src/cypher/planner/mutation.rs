//! Mutation planning: CREATE and DELETE.

use super::{LogicalOp, QueryPlanner};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};

impl QueryPlanner {
    /// Plan a CREATE statement
    pub(super) fn plan_create(&self, create: &CreateClause) -> CypherResult<LogicalOp> {
        if create.patterns.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty CREATE clause"));
        }

        // For now, support creating a single node or node-relationship-node
        let pattern = &create.patterns[0];

        if pattern.elements.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty pattern in CREATE"));
        }

        // Create single node
        if pattern.elements.len() == 1 {
            if let PatternElement::Node(node) = &pattern.elements[0] {
                return Ok(LogicalOp::CreateNode {
                    labels: node.labels.clone(),
                    properties: node.properties.clone().unwrap_or_default(),
                    variable: node.variable.clone(),
                });
            }
        }

        // Create node-relationship-node
        if pattern.elements.len() == 3 {
            let start_node = match &pattern.elements[0] {
                PatternElement::Node(n) => n,
                _ => return Err(CypherError::syntax(0, 0, "Expected node")),
            };
            let rel = match &pattern.elements[1] {
                PatternElement::Relationship(r) => r,
                _ => return Err(CypherError::syntax(0, 0, "Expected relationship")),
            };
            let end_node = match &pattern.elements[2] {
                PatternElement::Node(n) => n,
                _ => return Err(CypherError::syntax(0, 0, "Expected node")),
            };

            // For relationship creation, we need both nodes to have variables
            let from_var = start_node.variable.clone().ok_or_else(|| {
                CypherError::syntax(
                    0,
                    0,
                    "Start node must have a variable for relationship creation",
                )
            })?;
            let to_var = end_node.variable.clone().ok_or_else(|| {
                CypherError::syntax(
                    0,
                    0,
                    "End node must have a variable for relationship creation",
                )
            })?;

            // Get relationship type
            let rel_type = rel
                .rel_types
                .first()
                .ok_or_else(|| CypherError::syntax(0, 0, "Relationship must have a type"))?
                .clone();

            return Ok(LogicalOp::CreateRelationship {
                from: from_var,
                to: to_var,
                rel_type,
                properties: rel.properties.clone().unwrap_or_default(),
                variable: rel.variable.clone(),
            });
        }

        Err(CypherError::unsupported("Complex CREATE patterns"))
    }

    /// Plan a DELETE statement
    pub(super) fn plan_delete(
        &self,
        delete: &DeleteClause,
        input: Option<LogicalOp>,
    ) -> CypherResult<LogicalOp> {
        let input = input.unwrap_or(LogicalOp::SingleRow);

        // For each variable, create a delete operation
        // For simplicity, we'll delete the first variable as a node
        if let Some(var) = delete.variables.first() {
            Ok(LogicalOp::DeleteNode {
                input: Box::new(input),
                variable: var.clone(),
                detach: delete.detach,
            })
        } else {
            Err(CypherError::syntax(
                0,
                0,
                "DELETE requires at least one variable",
            ))
        }
    }

    /// Plan a query with mutations
    pub(super) fn plan_read_write(
        &self,
        query: &Query,
        mutations: &[Mutation],
    ) -> CypherResult<LogicalOp> {
        // First plan the query part
        let mut plan = self.plan_query(query)?;

        // Then apply mutations
        for mutation in mutations {
            plan = match mutation {
                Mutation::Create(create) => {
                    // Limitation: CREATE after MATCH currently replaces the plan
                    // rather than sequencing operations. Multi-clause mutations
                    // with data dependencies may produce incorrect results.
                    let create_plan = self.plan_create(create)?;
                    create_plan
                }
                Mutation::Delete(delete) => self.plan_delete(delete, Some(plan))?,
                Mutation::Set(_set) => {
                    return Err(CypherError::unsupported("SET clause"));
                }
            };
        }

        Ok(plan)
    }
}
