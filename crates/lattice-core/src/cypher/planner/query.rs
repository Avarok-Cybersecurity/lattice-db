//! Query planning: the clause pipeline (MATCH, OPTIONAL MATCH, UNWIND, WITH) and RETURN.
//!
//! Clauses are planned left to right over one plan. `scope` holds the variables the current
//! plan's rows bind, in binding order; `unit` is true while the plan is still the single empty
//! row a query starts from (so the first scan replaces it instead of joining with it).

use super::pattern::MatchState;
use super::projection::ProjectionKind;
use super::resolve::{resolve, Site};
use super::{LogicalOp, QueryPlanner};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};

impl QueryPlanner {
    /// Plan a read query
    pub(super) fn plan_query(&self, query: &Query) -> CypherResult<LogicalOp> {
        let mut plan = LogicalOp::SingleRow;
        let mut scope: Vec<String> = Vec::new();
        let mut unit = true;
        let mut anon = 0u32;

        for clause in &query.clauses {
            plan = match clause {
                Clause::Match(m) => self.plan_match(plan, &mut scope, unit, &mut anon, m)?,
                Clause::Unwind(u) => {
                    if scope.contains(&u.variable) {
                        return Err(CypherError::syntax(
                            0,
                            0,
                            format!("Variable `{}` already declared", u.variable),
                        ));
                    }
                    let expr = resolve(&u.expr, &scope, Site::Unwind)?;
                    scope.push(u.variable.clone());
                    LogicalOp::Unwind {
                        input: Box::new(plan),
                        expr,
                        variable: u.variable.clone(),
                    }
                }
                Clause::With(w) => {
                    let mut plan = self.plan_projection(
                        plan,
                        &mut scope,
                        &w.projection,
                        ProjectionKind::With,
                    )?;
                    if let Some(where_clause) = &w.where_clause {
                        plan = LogicalOp::Filter {
                            input: Box::new(plan),
                            predicate: resolve(&where_clause.predicate, &scope, Site::Where)?,
                        };
                    }
                    plan
                }
            };
            unit = false;
        }

        self.plan_projection(
            plan,
            &mut scope,
            &query.return_clause,
            ProjectionKind::Return,
        )
    }

    /// Plan MATCH / OPTIONAL MATCH and its WHERE over `input`
    fn plan_match(
        &self,
        input: LogicalOp,
        scope: &mut Vec<String>,
        unit: bool,
        anon: &mut u32,
        m: &MatchClause,
    ) -> CypherResult<LogicalOp> {
        if m.patterns.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty MATCH clause"));
        }
        let mut local_scope = scope.clone();
        // An OPTIONAL MATCH is planned over its input's rows (Argument) and joined back.
        let (mut plan, outer) = if m.optional {
            (LogicalOp::Argument, Some(input))
        } else {
            (input, None)
        };
        let mut st = MatchState {
            scope: &mut local_scope,
            unit: unit && outer.is_none(),
            anon,
            rels: Vec::new(),
        };
        for pattern in &m.patterns {
            plan = self.plan_pattern(plan, &mut st, pattern)?;
        }
        if let Some(where_clause) = &m.where_clause {
            plan = LogicalOp::Filter {
                input: Box::new(plan),
                predicate: resolve(&where_clause.predicate, &local_scope, Site::Where)?,
            };
        }
        *scope = local_scope;
        Ok(match outer {
            Some(input) => LogicalOp::OptionalMatch {
                input: Box::new(input),
                inner: Box::new(plan),
            },
            None => plan,
        })
    }
}
