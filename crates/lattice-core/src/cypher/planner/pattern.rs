//! Pattern planning: node scans, expansions and inline label/property constraints.
//!
//! A pattern's first node is scanned (by its first label) unless it is already bound; a scan
//! after other rows joins them as a cartesian product. Each relationship becomes an `Expand`
//! from the previous node. Labels and inline properties become a `Filter` right after the
//! operator that binds the node, so a filter directly over a label scan is pushed into it.

use super::resolve::{resolve, Site};
use super::{LogicalOp, QueryPlanner, MAX_PATH_LENGTH};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};

/// Planning state shared by the patterns of one MATCH clause
pub(super) struct MatchState<'a> {
    /// Variables bound so far (binding order)
    pub scope: &'a mut Vec<String>,
    /// The plan is still the query's initial single empty row
    pub unit: bool,
    /// Counter for hidden names of anonymous nodes and relationships
    pub anon: &'a mut u32,
    /// Single-hop relationships of this MATCH: each must bind a different relationship
    pub rels: Vec<String>,
}

impl MatchState<'_> {
    fn fresh(&mut self, kind: &str) -> String {
        *self.anon += 1;
        format!("#{kind}{}", self.anon)
    }

    fn bind(&mut self, var: &str) {
        if !self.scope.iter().any(|v| v == var) {
            self.scope.push(var.to_string());
        }
    }

    fn is_bound(&self, var: &str) -> bool {
        self.scope.iter().any(|v| v == var)
    }
}

/// `var:labels AND var.k = v ...` for the given constraints (empty when there are none)
fn constraints(
    var: &str,
    labels: &[String],
    props: Option<&MapLiteral>,
    scope: &[String],
) -> CypherResult<Vec<Expr>> {
    let mut preds = Vec::new();
    if !labels.is_empty() {
        preds.push(Expr::HasLabels {
            expr: Box::new(Expr::variable(var)),
            labels: labels.to_vec(),
        });
    }
    for (key, value) in props.map(|p| p.entries.as_slice()).unwrap_or_default() {
        let value = resolve(value, scope, Site::Pattern)?;
        preds.push(Expr::property(Expr::variable(var), key.as_str()).eq(value));
    }
    Ok(preds)
}

/// The same pattern read from its other end (relationship directions flipped)
fn reversed(pattern: &Pattern) -> Pattern {
    Pattern {
        elements: pattern
            .elements
            .iter()
            .rev()
            .map(|e| match e {
                PatternElement::Relationship(r) => {
                    let mut r = r.clone();
                    r.direction = match r.direction {
                        Direction::Outgoing => Direction::Incoming,
                        Direction::Incoming => Direction::Outgoing,
                        Direction::Both => Direction::Both,
                    };
                    PatternElement::Relationship(r)
                }
                node => node.clone(),
            })
            .collect(),
    }
}

/// A pattern whose first node is unbound but whose last node is bound is planned from the
/// bound end: the same matches, without a scan joined against every input row
fn oriented<'p>(pattern: &'p Pattern, scope: &[String]) -> std::borrow::Cow<'p, Pattern> {
    let bound = |e: Option<&PatternElement>| matches!(e, Some(PatternElement::Node(NodePattern { variable: Some(v), .. })) if scope.contains(v));
    if !bound(pattern.elements.first()) && bound(pattern.elements.last()) {
        std::borrow::Cow::Owned(reversed(pattern))
    } else {
        std::borrow::Cow::Borrowed(pattern)
    }
}

fn filtered(plan: LogicalOp, preds: Vec<Expr>) -> LogicalOp {
    match preds.into_iter().reduce(|a, b| a.and(b)) {
        Some(predicate) => LogicalOp::Filter {
            input: Box::new(plan),
            predicate,
        },
        None => plan,
    }
}

impl QueryPlanner {
    /// Plan one pattern on top of `plan`
    pub(super) fn plan_pattern(
        &self,
        mut plan: LogicalOp,
        st: &mut MatchState,
        pattern: &Pattern,
    ) -> CypherResult<LogicalOp> {
        let pattern = oriented(pattern, st.scope);
        let mut elements = pattern.elements.iter();
        let first = match elements.next() {
            Some(PatternElement::Node(n)) => n,
            _ => return Err(CypherError::syntax(0, 0, "Pattern must start with a node")),
        };

        let mut prev = first.variable.clone().unwrap_or_else(|| st.fresh("anon"));
        if st.is_bound(&prev) {
            let preds = constraints(&prev, &first.labels, first.properties.as_ref(), st.scope)?;
            plan = filtered(plan, preds);
        } else {
            let scan = match first.labels.first() {
                Some(label) => LogicalOp::NodeByLabelScan {
                    variable: prev.clone(),
                    label: label.clone(),
                    predicate: None,
                },
                None => LogicalOp::AllNodesScan {
                    variable: prev.clone(),
                },
            };
            plan = if st.unit {
                scan
            } else {
                LogicalOp::Cartesian {
                    left: Box::new(plan),
                    right: Box::new(scan),
                }
            };
            st.unit = false;
            st.bind(&prev);
            let rest = first.labels.get(1..).unwrap_or_default();
            let preds = constraints(&prev, rest, first.properties.as_ref(), st.scope)?;
            plan = filtered(plan, preds);
        }

        while let Some(element) = elements.next() {
            let (rel, node) = match (element, elements.next()) {
                (PatternElement::Relationship(r), Some(PatternElement::Node(n))) => (r, n),
                _ => {
                    return Err(CypherError::syntax(
                        0,
                        0,
                        "Incomplete pattern: relationship without target node",
                    ))
                }
            };
            let to = node.variable.clone().unwrap_or_else(|| st.fresh("anon"));
            let (min_hops, max_hops, rel_variable) = match &rel.length {
                Some(len) => {
                    if let Some(r) = &rel.variable {
                        return Err(CypherError::unsupported(format!(
                            "binding a variable-length relationship to a variable ('{r}')"
                        )));
                    }
                    if rel.properties.is_some() {
                        return Err(CypherError::unsupported(
                            "properties on a variable-length relationship",
                        ));
                    }
                    let min = len.min.unwrap_or(1);
                    let max = len.max.unwrap_or(MAX_PATH_LENGTH).min(MAX_PATH_LENGTH);
                    (min, max, None)
                }
                None => {
                    let r = rel.variable.clone().unwrap_or_else(|| st.fresh("rel"));
                    (1, 1, Some(r))
                }
            };
            plan = LogicalOp::Expand {
                input: Box::new(plan),
                from: prev,
                rel_variable: rel_variable.clone(),
                to: to.clone(),
                rel_types: rel.rel_types.clone(),
                direction: rel.direction,
                min_hops,
                max_hops,
                distinct_from: match rel_variable {
                    Some(_) => st.rels.clone(),
                    None => Vec::new(),
                },
            };
            st.bind(&to);
            let mut preds = constraints(&to, &node.labels, node.properties.as_ref(), st.scope)?;
            if let Some(r) = &rel_variable {
                st.bind(r);
                st.rels.push(r.clone());
                preds.extend(constraints(r, &[], rel.properties.as_ref(), st.scope)?);
            }
            plan = filtered(plan, preds);
            prev = to;
        }

        Ok(plan)
    }
}
