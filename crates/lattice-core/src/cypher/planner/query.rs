//! Query planning: MATCH, OPTIONAL MATCH, WHERE and the projection pipeline.

use super::{LogicalOp, QueryPlanner};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};

impl QueryPlanner {
    /// Plan a query
    pub(super) fn plan_query(&self, query: &Query) -> CypherResult<LogicalOp> {
        // Start with the MATCH clause or a single row
        let mut plan = if let Some(match_clause) = &query.match_clause {
            self.plan_match(match_clause)?
        } else {
            LogicalOp::SingleRow
        };

        // Apply OPTIONAL MATCH clauses
        for optional in &query.optional_matches {
            plan = self.plan_optional_match(plan, optional)?;
        }

        // Apply WHERE clause
        if let Some(where_clause) = &query.where_clause {
            plan = LogicalOp::Filter {
                input: Box::new(plan),
                predicate: where_clause.predicate.clone(),
            };
        }

        // Apply ORDER BY (with optional limit for partial sort optimization)
        // When there's no SKIP, we can pass the LIMIT to Sort for O(n) partial sort
        let sort_limit = if query.skip.is_none() {
            query.limit
        } else {
            None
        };
        if let Some(order_by) = &query.order_by {
            plan = LogicalOp::Sort {
                input: Box::new(plan),
                items: order_by.items.clone(),
                limit: sort_limit,
            };
        }

        // Apply SKIP
        if let Some(skip) = query.skip {
            plan = LogicalOp::Skip {
                input: Box::new(plan),
                count: skip,
            };
        }

        // Apply LIMIT (even if Sort already handled it, keeps semantics correct)
        if let Some(limit) = query.limit {
            plan = LogicalOp::Limit {
                input: Box::new(plan),
                count: limit,
            };
        }

        // Apply RETURN clause (projection)
        if query.return_clause.distinct {
            plan = LogicalOp::Distinct {
                input: Box::new(plan),
            };
        }

        plan = LogicalOp::Project {
            input: Box::new(plan),
            items: query.return_clause.items.clone(),
        };

        Ok(plan)
    }

    /// Plan a MATCH clause
    pub(super) fn plan_match(&self, match_clause: &MatchClause) -> CypherResult<LogicalOp> {
        if match_clause.patterns.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty MATCH clause"));
        }

        // Limitation: only single-pattern MATCH is supported.
        // Multi-pattern MATCH (Cartesian product) requires cross-join planning.
        if match_clause.patterns.len() > 1 {
            return Err(CypherError::unsupported(
                "Multiple MATCH patterns (Cartesian product) not yet supported",
            ));
        }
        let pattern = &match_clause.patterns[0];
        self.plan_pattern(pattern)
    }

    /// Plan an OPTIONAL MATCH clause
    pub(super) fn plan_optional_match(
        &self,
        input: LogicalOp,
        match_clause: &MatchClause,
    ) -> CypherResult<LogicalOp> {
        // Limitation: OPTIONAL MATCH currently behaves like MATCH (inner join).
        // Proper left-outer-join semantics (NULL rows for non-matches) not yet implemented.

        if match_clause.patterns.is_empty() {
            return Ok(input);
        }

        let pattern = &match_clause.patterns[0];

        // Only support patterns that start with an existing variable
        if let Some(PatternElement::Node(node)) = pattern.elements.first() {
            if let Some(ref _var) = node.variable {
                // Expand from existing variable
                return self.plan_pattern_expansion(input, pattern);
            }
        }

        // For standalone patterns, just plan them
        self.plan_pattern(pattern)
    }
}
