//! Pattern planning: node scans, expansions and inline property/label filters.

use super::{LogicalOp, QueryPlanner};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};

use super::MAX_PATH_LENGTH;

impl QueryPlanner {
    /// Plan a single pattern
    pub(super) fn plan_pattern(&self, pattern: &Pattern) -> CypherResult<LogicalOp> {
        if pattern.elements.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty pattern"));
        }

        // Get the first node
        let first_node = match &pattern.elements[0] {
            PatternElement::Node(n) => n,
            _ => return Err(CypherError::syntax(0, 0, "Pattern must start with a node")),
        };

        // Plan the initial scan
        let mut plan = self.plan_node_scan(first_node)?;

        // Process remaining elements (rel-node pairs)
        let mut i = 1;
        while i < pattern.elements.len() {
            if i + 1 >= pattern.elements.len() {
                return Err(CypherError::syntax(
                    0,
                    0,
                    "Incomplete pattern: relationship without target node",
                ));
            }

            let rel = match &pattern.elements[i] {
                PatternElement::Relationship(r) => r,
                _ => {
                    return Err(CypherError::syntax(
                        0,
                        0,
                        "Expected relationship after node",
                    ))
                }
            };

            let target_node = match &pattern.elements[i + 1] {
                PatternElement::Node(n) => n,
                _ => {
                    return Err(CypherError::syntax(
                        0,
                        0,
                        "Expected node after relationship",
                    ))
                }
            };

            // Get the source variable name
            let from_var = self.get_last_node_variable(&plan)?;

            // Get or generate target variable
            let to_var = target_node
                .variable
                .clone()
                .unwrap_or_else(|| String::from(format!("_anon_{}", i)));

            // Determine hop range (capped at MAX_PATH_LENGTH for DoS protection)
            let (min_hops, max_hops) = if let Some(len) = &rel.length {
                let min = len.min.unwrap_or(1);
                let max = len.max.unwrap_or(MAX_PATH_LENGTH).min(MAX_PATH_LENGTH);
                (min, max)
            } else {
                (1, 1)
            };

            // Create expand operation
            plan = LogicalOp::Expand {
                input: Box::new(plan),
                from: from_var,
                rel_variable: rel.variable.clone(),
                to: to_var.clone(),
                rel_types: rel.rel_types.clone(),
                direction: rel.direction,
                min_hops,
                max_hops,
            };

            // Apply target node constraints (labels, properties)
            if !target_node.labels.is_empty() || target_node.properties.is_some() {
                let filter = self.build_node_filter(&to_var, target_node)?;
                plan = LogicalOp::Filter {
                    input: Box::new(plan),
                    predicate: filter,
                };
            }

            i += 2;
        }

        Ok(plan)
    }

    /// Plan pattern expansion from existing input
    pub(super) fn plan_pattern_expansion(
        &self,
        input: LogicalOp,
        pattern: &Pattern,
    ) -> CypherResult<LogicalOp> {
        // Similar to plan_pattern but starts from existing input
        // Skip the first node if it's a variable reference

        if pattern.elements.len() < 3 {
            return Ok(input);
        }

        let first_node = match &pattern.elements[0] {
            PatternElement::Node(n) => n,
            _ => return Err(CypherError::syntax(0, 0, "Pattern must start with a node")),
        };

        let from_var = first_node
            .variable
            .clone()
            .ok_or_else(|| CypherError::syntax(0, 0, "OPTIONAL MATCH requires bound variable"))?;

        let mut plan = input;
        let mut i = 1;

        while i < pattern.elements.len() {
            if i + 1 >= pattern.elements.len() {
                break;
            }

            let rel = match &pattern.elements[i] {
                PatternElement::Relationship(r) => r,
                _ => break,
            };

            let target_node = match &pattern.elements[i + 1] {
                PatternElement::Node(n) => n,
                _ => break,
            };

            let to_var = target_node
                .variable
                .clone()
                .unwrap_or_else(|| String::from(format!("_anon_{}", i)));

            // Compute from variable before moving plan
            let from = if i == 1 {
                from_var.clone()
            } else {
                self.get_last_node_variable(&plan)?
            };

            plan = LogicalOp::OptionalExpand {
                input: Box::new(plan),
                from,
                rel_variable: rel.variable.clone(),
                to: to_var.clone(),
                rel_types: rel.rel_types.clone(),
                direction: rel.direction,
            };

            i += 2;
        }

        Ok(plan)
    }

    /// Plan a node scan
    pub(super) fn plan_node_scan(&self, node: &NodePattern) -> CypherResult<LogicalOp> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| String::from("_anon_0"));

        // Choose scan type based on constraints
        let mut plan = if node.labels.is_empty() {
            LogicalOp::AllNodesScan {
                variable: variable.clone(),
            }
        } else {
            // Use label scan for first label
            LogicalOp::NodeByLabelScan {
                variable: variable.clone(),
                label: node.labels[0].clone(),
                predicate: None,
            }
        };

        // Add filter for additional labels
        if node.labels.len() > 1 {
            let mut predicates = Vec::new();
            for label in node.labels.iter().skip(1) {
                // Check if node has this label
                predicates.push(Expr::function(
                    "hasLabel",
                    vec![
                        Expr::variable(variable.clone()),
                        Expr::literal(label.as_str()),
                    ],
                ));
            }
            // SAFETY: predicates is non-empty because labels.len() > 1 and skip(1) yields at least 1
            if let Some(combined) = predicates.into_iter().reduce(|a, b| a.and(b)) {
                plan = LogicalOp::Filter {
                    input: Box::new(plan),
                    predicate: combined,
                };
            }
        }

        // Add property filter
        if let Some(props) = &node.properties {
            let filter = self.build_property_filter(&variable, props)?;
            plan = LogicalOp::Filter {
                input: Box::new(plan),
                predicate: filter,
            };
        }

        Ok(plan)
    }

    /// Build a filter expression for node constraints
    pub(super) fn build_node_filter(
        &self,
        variable: &str,
        node: &NodePattern,
    ) -> CypherResult<Expr> {
        let mut predicates = Vec::new();

        // Label checks
        for label in &node.labels {
            predicates.push(Expr::function(
                "hasLabel",
                vec![Expr::variable(variable), Expr::literal(label.as_str())],
            ));
        }

        // Property checks
        if let Some(props) = &node.properties {
            for (key, value) in &props.entries {
                predicates.push(
                    Expr::property(Expr::variable(variable), key.as_str()).eq((**value).clone()),
                );
            }
        }

        // Combine predicates with AND
        predicates
            .into_iter()
            .reduce(|a, b| a.and(b))
            .ok_or_else(|| CypherError::internal("Empty filter"))
    }

    /// Build a filter expression for property constraints
    pub(super) fn build_property_filter(
        &self,
        variable: &str,
        props: &MapLiteral,
    ) -> CypherResult<Expr> {
        let mut predicates = Vec::new();

        for (key, value) in &props.entries {
            predicates
                .push(Expr::property(Expr::variable(variable), key.as_str()).eq((**value).clone()));
        }

        predicates
            .into_iter()
            .reduce(|a, b| a.and(b))
            .ok_or_else(|| CypherError::internal("Empty property filter"))
    }

    /// Get the variable name of the last node in a plan
    pub(super) fn get_last_node_variable(&self, plan: &LogicalOp) -> CypherResult<String> {
        match plan {
            LogicalOp::AllNodesScan { variable } => Ok(variable.clone()),
            LogicalOp::NodeByLabelScan { variable, .. } => Ok(variable.clone()),
            LogicalOp::NodeByIdSeek { variable, .. } => Ok(variable.clone()),
            LogicalOp::NodeByPropertySeek { variable, .. } => Ok(variable.clone()),
            LogicalOp::Expand { to, .. } => Ok(to.clone()),
            LogicalOp::OptionalExpand { to, .. } => Ok(to.clone()),
            LogicalOp::Filter { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Project { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Sort { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Skip { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Limit { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Distinct { input } => self.get_last_node_variable(input),
            _ => Err(CypherError::internal("Cannot determine node variable")),
        }
    }
}
