//! Canonical keys for DISTINCT, grouping and aggregate DISTINCT.
//!
//! Two values get the same key exactly when openCypher treats them as the same grouping value
//! (equivalence): null groups with null, `1` with `1.0`, NaN with NaN, maps regardless of key
//! order. The key derives `Hash` + `Eq`, so deduplication is a plain hash set with no
//! collision handling of its own.

use super::compare::exact_int;
use crate::types::value::CypherValue;
use crate::types::SharedStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Canon {
    Null,
    Bool(bool),
    Int(i64),
    /// A non-integral float, by bit pattern (NaN normalised)
    Float(u64),
    Str(SharedStr),
    List(Vec<Canon>),
    Map(Vec<(String, Canon)>),
    Node(u64),
    Rel(u64),
    /// Temporal, spatial, bytes and paths: by representation
    Other(String),
}

impl Canon {
    pub(crate) fn of(v: &CypherValue) -> Canon {
        match v {
            CypherValue::Null => Canon::Null,
            CypherValue::Bool(b) => Canon::Bool(*b),
            CypherValue::Int(i) => Canon::Int(*i),
            CypherValue::Float(f) => {
                if f.is_nan() {
                    Canon::Float(f64::NAN.to_bits())
                } else if let Some(i) = exact_int(*f) {
                    Canon::Int(i)
                } else {
                    Canon::Float(f.to_bits())
                }
            }
            CypherValue::String(s) => Canon::Str(s.clone()),
            CypherValue::List(items) => Canon::List(items.iter().map(Canon::of).collect()),
            CypherValue::Map(entries) => {
                let mut m: Vec<(String, Canon)> = entries
                    .iter()
                    .map(|(k, v)| (k.clone(), Canon::of(v)))
                    .collect();
                m.sort_by(|a, b| a.0.cmp(&b.0));
                Canon::Map(m)
            }
            CypherValue::NodeRef(id) => Canon::Node(*id),
            CypherValue::RelationshipRef(id) => Canon::Rel(*id),
            other => Canon::Other(format!("{other:?}")),
        }
    }

    /// The key of a whole row
    pub(crate) fn row(values: &[CypherValue]) -> Vec<Canon> {
        values.iter().map(Canon::of).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use CypherValue as V;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn equivalent_values_share_a_key_and_different_ones_do_not() {
        assert_eq!(Canon::of(&V::Int(1)), Canon::of(&V::Float(1.0)));
        assert_eq!(
            Canon::of(&V::Float(f64::NAN)),
            Canon::of(&V::Float(-f64::NAN))
        );
        assert_eq!(
            Canon::of(&V::map_from([("a", V::Int(1)), ("b", V::Int(2))])),
            Canon::of(&V::map_from([("b", V::Int(2)), ("a", V::Int(1))]))
        );
        assert_ne!(Canon::of(&V::Int(1)), Canon::of(&V::string("1")));
        assert_ne!(Canon::of(&V::Float(1.5)), Canon::of(&V::Int(1)));
        assert_ne!(Canon::of(&V::NodeRef(1)), Canon::of(&V::RelationshipRef(1)));
        assert_ne!(Canon::of(&V::Null), Canon::of(&V::Bool(false)));
    }
}
