//! Value access: `x.prop`, `x[i]`, `x[a..b]` on nodes, relationships, maps and lists.

use super::{ExecutionContext, QueryExecutor};
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::point::Point;
use crate::types::value::CypherValue;

/// A list position: negative counts from the end; None when out of range
fn position(len: usize, i: i64) -> Option<usize> {
    let len = i64::try_from(len).ok()?;
    let p = if i < 0 { len + i } else { i };
    (0..len).contains(&p).then_some(p as usize)
}

/// A slice bound clamped into 0..=len
fn bound(len: usize, i: i64) -> usize {
    let len_i = i64::try_from(len).unwrap_or(i64::MAX);
    let p = if i < 0 { len_i.saturating_add(i) } else { i };
    p.clamp(0, len_i) as usize
}

impl QueryExecutor {
    /// `base.property`
    pub(crate) fn property_of(
        &self,
        base: &CypherValue,
        property: &str,
        ctx: &ExecutionContext,
    ) -> CypherResult<CypherValue> {
        match base {
            CypherValue::Null => Ok(CypherValue::Null),
            CypherValue::NodeRef(id) => {
                if let Some(cached) = ctx.get_property_cached(*id, property) {
                    return Ok(cached);
                }
                match ctx.get_point_cached(*id) {
                    Some(point) => {
                        let value = self.get_point_property(&point, property)?;
                        ctx.cache_property(*id, property, value.clone());
                        Ok(value)
                    }
                    None => Ok(CypherValue::Null),
                }
            }
            CypherValue::RelationshipRef(h) => {
                let rel = ctx.rel(*h)?;
                Ok(match property {
                    "weight" => CypherValue::Float(f64::from(rel.weight)),
                    _ => CypherValue::Null,
                })
            }
            CypherValue::Map(entries) => Ok(entries
                .iter()
                .find(|(k, _)| k == property)
                .map_or(CypherValue::Null, |(_, v)| v.clone())),
            other => Err(CypherError::invalid("property access", other.type_name())),
        }
    }

    /// `base[index]`
    pub(crate) fn index_value(
        &self,
        base: &CypherValue,
        index: &CypherValue,
        ctx: &ExecutionContext,
    ) -> CypherResult<CypherValue> {
        match (base, index) {
            (CypherValue::Null, _) | (_, CypherValue::Null) => Ok(CypherValue::Null),
            (CypherValue::List(items), CypherValue::Int(i)) => {
                Ok(position(items.len(), *i).map_or(CypherValue::Null, |p| items[p].clone()))
            }
            (CypherValue::List(_), other) => {
                Err(CypherError::invalid("list index", other.type_name()))
            }
            (
                CypherValue::Map(_) | CypherValue::NodeRef(_) | CypherValue::RelationshipRef(_),
                CypherValue::String(key),
            ) => self.property_of(base, key, ctx),
            (
                CypherValue::Map(_) | CypherValue::NodeRef(_) | CypherValue::RelationshipRef(_),
                other,
            ) => Err(CypherError::invalid("map key", other.type_name())),
            (other, _) => Err(CypherError::invalid("subscript", other.type_name())),
        }
    }

    /// `base[from..to]` (a missing bound is the start / end; a null bound gives null)
    pub(crate) fn slice_value(
        &self,
        base: &CypherValue,
        from: Option<&CypherValue>,
        to: Option<&CypherValue>,
    ) -> CypherResult<CypherValue> {
        let items = match base {
            CypherValue::Null => return Ok(CypherValue::Null),
            CypherValue::List(items) => items,
            other => return Err(CypherError::invalid("slice", other.type_name())),
        };
        let resolve = |b: Option<&CypherValue>, default: usize| match b {
            None => Ok(Some(default)),
            Some(CypherValue::Null) => Ok(None),
            Some(CypherValue::Int(i)) => Ok(Some(bound(items.len(), *i))),
            Some(other) => Err(CypherError::invalid("slice bound", other.type_name())),
        };
        match (resolve(from, 0)?, resolve(to, items.len())?) {
            (Some(a), Some(b)) => Ok(CypherValue::List(
                items
                    .get(a..b.max(a))
                    .map(<[CypherValue]>::to_vec)
                    .unwrap_or_default(),
            )),
            _ => Ok(CypherValue::Null),
        }
    }

    /// Get a property from a point
    ///
    /// # Security
    /// Properties starting with `_` (except `_labels`) are internal and blocked
    /// to prevent access to internal metadata.
    pub(crate) fn get_point_property(
        &self,
        point: &Point,
        property: &str,
    ) -> CypherResult<CypherValue> {
        match property {
            "id" => return Ok(CypherValue::Int(point.id as i64)),
            "_labels" => return Ok(CypherValue::list(self.point_labels(point)?)),
            _ => {}
        }

        // Block access to other internal properties (starting with _)
        if property.starts_with('_') {
            return Err(CypherError::UnknownProperty {
                variable: "node".to_string(),
                property: property.to_string(),
            });
        }

        match point.payload.get(property) {
            Some(value_bytes) => self.json_bytes_to_cypher_value(value_bytes),
            None => Ok(CypherValue::Null),
        }
    }

    /// A point's labels, in stored order
    pub(crate) fn point_labels(&self, point: &Point) -> CypherResult<Vec<CypherValue>> {
        match point.payload.get("_labels") {
            Some(bytes) => serde_json::from_slice::<Vec<String>>(bytes)
                .map(|labels| labels.into_iter().map(CypherValue::from).collect())
                .map_err(|e| CypherError::internal(format!("node {}: bad _labels: {e}", point.id))),
            None => Ok(Vec::new()),
        }
    }

    /// A point's user properties (internal `_` keys hidden), sorted by key
    pub(crate) fn point_properties(
        &self,
        point: &Point,
    ) -> CypherResult<Vec<(String, CypherValue)>> {
        let mut keys: Vec<&String> = point
            .payload
            .keys()
            .filter(|k| !k.starts_with('_'))
            .collect();
        keys.sort();
        keys.into_iter()
            .map(|k| {
                Ok((
                    k.clone(),
                    self.json_bytes_to_cypher_value(&point.payload[k])?,
                ))
            })
            .collect()
    }
}
