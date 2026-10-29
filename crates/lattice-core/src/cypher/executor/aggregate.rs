//! Aggregation: group rows by key values, fold each group with accumulators.
//!
//! Groups are emitted in first-seen order and each accumulator sees its group's rows in input
//! order, so `collect()` after an ORDER BY keeps that order. With no grouping keys there is
//! exactly one group, even over zero rows (`count(*)` = 0, `collect()` = []).

use super::canon::Canon;
use super::compare::order_cmp;
use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats};
use crate::cypher::ast::{AggregateKind, Expr, ProjectionItem};
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::{AggregateCall, LogicalOp};
use crate::cypher::row::{ExecutorRow, Row};
use crate::types::value::CypherValue;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Running state of one aggregate over one group
enum Acc {
    Count(i64),
    Collect(Vec<CypherValue>),
    Extreme(Option<CypherValue>),
    SumInt(i64),
    SumFloat(f64),
    Avg { sum: f64, n: i64 },
}

struct Accumulator {
    acc: Acc,
    /// DISTINCT: canonical keys of the values already folded in
    seen: Option<HashSet<Canon>>,
}

impl Accumulator {
    fn new(call: &AggregateCall) -> Self {
        let acc = match call.kind {
            AggregateKind::Count => Acc::Count(0),
            AggregateKind::Collect => Acc::Collect(Vec::new()),
            AggregateKind::Min | AggregateKind::Max => Acc::Extreme(None),
            AggregateKind::Sum => Acc::SumInt(0),
            AggregateKind::Avg => Acc::Avg { sum: 0.0, n: 0 },
        };
        Self {
            acc,
            seen: call.distinct.then(HashSet::new),
        }
    }

    /// Fold one row's argument value in (`None` is a `count(*)` row)
    fn update(&mut self, kind: AggregateKind, value: Option<CypherValue>) -> CypherResult<()> {
        let v = match value {
            None => {
                if let Acc::Count(n) = &mut self.acc {
                    *n += 1;
                }
                return Ok(());
            }
            Some(CypherValue::Null) => return Ok(()),
            Some(v) => v,
        };
        if let Some(seen) = &mut self.seen {
            if !seen.insert(Canon::of(&v)) {
                return Ok(());
            }
        }
        let number = match &v {
            CypherValue::Int(i) => Some((Some(*i), *i as f64)),
            CypherValue::Float(f) => Some((None, *f)),
            _ => None,
        };
        let type_name = v.type_name();
        let numeric = |kind: AggregateKind| {
            number.ok_or_else(|| CypherError::invalid(format!("{}()", kind.name()), type_name))
        };
        let next = match &mut self.acc {
            Acc::Count(n) => {
                *n += 1;
                None
            }
            Acc::Collect(items) => {
                items.push(v);
                None
            }
            Acc::Extreme(best) => {
                let want = if kind == AggregateKind::Min {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
                if best.as_ref().map_or(true, |b| order_cmp(&v, b) == want) {
                    *best = Some(v);
                }
                None
            }
            Acc::SumInt(total) => match numeric(kind)? {
                (Some(i), _) => {
                    *total =
                        total
                            .checked_add(i)
                            .ok_or_else(|| CypherError::ArithmeticOverflow {
                                operation: "sum()".to_string(),
                            })?;
                    None
                }
                // The first FLOAT switches the sum to FLOAT.
                (None, f) => Some(Acc::SumFloat(*total as f64 + f)),
            },
            Acc::SumFloat(total) => {
                *total += numeric(kind)?.1;
                None
            }
            Acc::Avg { sum, n } => {
                *sum += numeric(kind)?.1;
                *n += 1;
                None
            }
        };
        if let Some(acc) = next {
            self.acc = acc;
        }
        Ok(())
    }

    fn finish(self) -> CypherValue {
        match self.acc {
            Acc::Count(n) => CypherValue::Int(n),
            Acc::Collect(items) => CypherValue::List(items),
            Acc::Extreme(best) => best.unwrap_or(CypherValue::Null),
            Acc::SumInt(t) => CypherValue::Int(t),
            Acc::SumFloat(t) => CypherValue::Float(t),
            Acc::Avg { n: 0, .. } => CypherValue::Null,
            Acc::Avg { sum, n } => CypherValue::Float(sum / n as f64),
        }
    }
}

impl QueryExecutor {
    /// Execute an Aggregate operation
    pub(crate) fn execute_aggregate(
        &self,
        input: &LogicalOp,
        keys: &[Expr],
        aggregates: &[AggregateCall],
        items: &[ProjectionItem],
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (rows, stats) = self.execute_op(input, ctx)?;

        let mut index: HashMap<Vec<Canon>, usize> = HashMap::new();
        let mut groups: Vec<(Vec<CypherValue>, Vec<Accumulator>)> = Vec::new();
        for row in &rows {
            let key = keys
                .iter()
                .map(|k| self.evaluate_expr(k, row, ctx))
                .collect::<CypherResult<Vec<_>>>()?;
            let g = *index.entry(Canon::row(&key)).or_insert_with(|| {
                groups.push((key, aggregates.iter().map(Accumulator::new).collect()));
                groups.len() - 1
            });
            for (acc, call) in groups[g].1.iter_mut().zip(aggregates) {
                let value = call
                    .arg
                    .as_ref()
                    .map(|a| self.evaluate_expr(a, row, ctx))
                    .transpose()?;
                acc.update(call.kind, value)?;
            }
        }
        if groups.is_empty() && keys.is_empty() {
            groups.push((
                Vec::new(),
                aggregates.iter().map(Accumulator::new).collect(),
            ));
        }

        // Items read `#key{i}` / `#agg{j}` from [key values..., aggregate results...].
        ctx.rebind(
            (0..keys.len())
                .map(|i| format!("#key{i}"))
                .chain((0..aggregates.len()).map(|j| format!("#agg{j}"))),
        );
        let mut out: InternalRows = Vec::with_capacity(groups.len());
        for (key, accs) in groups {
            let mut group_row = key;
            group_row.extend(accs.into_iter().map(Accumulator::finish));
            let mut projected = ExecutorRow::with_capacity(items.len());
            for item in items {
                Row::push(
                    &mut projected,
                    self.evaluate_expr(&item.expr, &group_row, ctx)?,
                );
            }
            out.push(projected);
        }
        ctx.rebind(items.iter().map(|i| i.alias.clone().unwrap_or_default()));
        Ok((out, stats))
    }
}
