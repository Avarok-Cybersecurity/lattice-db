//! openCypher value semantics: equality, comparison, three-valued logic, orderability.
//!
//! - `equals` is `=`: null when either side is null (or a list/map holds a null that decides
//!   it), numbers compare across INTEGER/FLOAT, values of different types are unequal.
//! - `compare` is `<`, `<=`, `>`, `>=`: null for null or for values of incomparable types.
//! - `order_cmp` is the total order ORDER BY, min() and max() use: MAP < NODE < RELATIONSHIP
//!   < LIST < PATH < (temporal, spatial, bytes) < STRING < BOOLEAN < NUMBER < NULL, so nulls
//!   sort last ascending and first descending.

use crate::types::value::CypherValue;
use std::cmp::Ordering;

/// Floats inside the i64 range (-2^63 <= f < 2^63); their integral part is an exact i64
pub(crate) const I64_RANGE: std::ops::Range<f64> =
    -9.223_372_036_854_776e18..9.223_372_036_854_776e18;

/// The integer a float equals exactly, if any
pub(crate) fn exact_int(f: f64) -> Option<i64> {
    (f.fract() == 0.0 && I64_RANGE.contains(&f)).then_some(f as i64)
}

/// Exact INTEGER = FLOAT (no rounding of large integers)
fn int_float_eq(i: i64, f: f64) -> bool {
    exact_int(f) == Some(i)
}

/// Exact INTEGER vs FLOAT ordering (None when the float is NaN); `i as f64` would round
/// integers beyond 2^53 onto their float neighbours
fn int_float_cmp(i: i64, f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    if f >= I64_RANGE.end {
        return Some(Ordering::Less);
    }
    if f < I64_RANGE.start {
        return Some(Ordering::Greater);
    }
    // |f| < 2^63, so its integral part is an exact i64.
    Some(i.cmp(&(f.trunc() as i64)).then_with(|| {
        if f.fract() > 0.0 {
            Ordering::Less
        } else if f.fract() < 0.0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }))
}

/// Combine element-wise equalities: any false decides, else any null makes it null
fn all_equal(pairs: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut saw_null = false;
    for p in pairs {
        match p {
            Some(false) => return Some(false),
            None => saw_null = true,
            Some(true) => {}
        }
    }
    if saw_null {
        None
    } else {
        Some(true)
    }
}

/// openCypher `=`
pub(crate) fn equals(a: &CypherValue, b: &CypherValue) -> Option<bool> {
    use CypherValue::*;
    match (a, b) {
        (Null, _) | (_, Null) => None,
        (Int(x), Int(y)) => Some(x == y),
        (Float(x), Float(y)) => Some(x == y),
        (Int(i), Float(f)) | (Float(f), Int(i)) => Some(int_float_eq(*i, *f)),
        (List(x), List(y)) => {
            if x.len() != y.len() {
                return Some(false);
            }
            all_equal(x.iter().zip(y).map(|(p, q)| equals(p, q)))
        }
        (Map(x), Map(y)) => {
            if x.len() != y.len() {
                return Some(false);
            }
            all_equal(
                x.iter()
                    .map(|(k, v)| match y.iter().find(|(k2, _)| k2 == k) {
                        Some((_, w)) => equals(v, w),
                        None => Some(false),
                    }),
            )
        }
        (String(x), String(y)) => Some(x == y),
        (Bool(x), Bool(y)) => Some(x == y),
        (NodeRef(x), NodeRef(y)) | (RelationshipRef(x), RelationshipRef(y)) => Some(x == y),
        _ if std::mem::discriminant(a) == std::mem::discriminant(b) => Some(a == b),
        _ => Some(false),
    }
}

/// openCypher `<` family: the ordering of two comparable values, `Err(())` for null or
/// incomparable values, `Ok(None)` for unordered numbers (NaN)
pub(crate) fn compare(a: &CypherValue, b: &CypherValue) -> Result<Option<Ordering>, ()> {
    use CypherValue::*;
    match (a, b) {
        (Int(x), Int(y)) => Ok(Some(x.cmp(y))),
        (Float(x), Float(y)) => Ok(x.partial_cmp(y)),
        (Int(i), Float(f)) => Ok(int_float_cmp(*i, *f)),
        (Float(f), Int(i)) => Ok(int_float_cmp(*i, *f).map(Ordering::reverse)),
        (String(x), String(y)) => Ok(Some(x.cmp(y))),
        (Bool(x), Bool(y)) => Ok(Some(x.cmp(y))),
        (List(x), List(y)) => {
            for (p, q) in x.iter().zip(y) {
                match compare(p, q)? {
                    Some(Ordering::Equal) => continue,
                    other => return Ok(other),
                }
            }
            Ok(Some(x.len().cmp(&y.len())))
        }
        _ => Err(()),
    }
}

/// Three-valued AND
pub(crate) fn and3(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// Three-valued OR
pub(crate) fn or3(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// Three-valued XOR
pub(crate) fn xor3(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    Some(a? ^ b?)
}

/// A boolean result as a value (null for None)
pub(crate) fn tri(v: Option<bool>) -> CypherValue {
    v.map_or(CypherValue::Null, CypherValue::Bool)
}

/// Position of a value's type in the ORDER BY total order
fn type_rank(v: &CypherValue) -> u8 {
    use CypherValue::*;
    match v {
        Map(_) => 0,
        NodeRef(_) => 1,
        RelationshipRef(_) => 2,
        List(_) => 3,
        Path(_) => 4,
        Date { .. } => 5,
        Time { .. } => 6,
        DateTime { .. } => 7,
        Duration { .. } => 8,
        Point2D { .. } | Point3D { .. } => 9,
        Bytes(_) => 10,
        String(_) => 11,
        Bool(_) => 12,
        Int(_) | Float(_) => 13,
        Null => 14,
    }
}

/// The total order ORDER BY uses (see module docs); NaN sorts above every other number
pub(crate) fn order_cmp(a: &CypherValue, b: &CypherValue) -> Ordering {
    use CypherValue::*;
    let rank = type_rank(a).cmp(&type_rank(b));
    if rank != Ordering::Equal {
        return rank;
    }
    let num = |x: &CypherValue| match x {
        Int(i) => (*i as f64, Some(*i)),
        Float(f) => (*f, None),
        _ => (0.0, None),
    };
    match (a, b) {
        (Int(x), Int(y)) => x.cmp(y),
        (Int(_) | Float(_), Int(_) | Float(_)) => {
            let ((fa, _), (fb, _)) = (num(a), num(b));
            match (fa.is_nan(), fb.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => match (a, b) {
                    (Int(i), Float(f)) => int_float_cmp(*i, *f).unwrap_or(Ordering::Equal),
                    (Float(f), Int(i)) => int_float_cmp(*i, *f)
                        .map(Ordering::reverse)
                        .unwrap_or(Ordering::Equal),
                    _ => fa.partial_cmp(&fb).unwrap_or(Ordering::Equal),
                },
            }
        }
        (String(x), String(y)) => x.cmp(y),
        (Bool(x), Bool(y)) => x.cmp(y),
        (NodeRef(x), NodeRef(y)) | (RelationshipRef(x), RelationshipRef(y)) => x.cmp(y),
        (List(x), List(y)) => x
            .iter()
            .zip(y)
            .map(|(p, q)| order_cmp(p, q))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or_else(|| x.len().cmp(&y.len())),
        (Map(x), Map(y)) => {
            fn sorted(
                m: &[(std::string::String, CypherValue)],
            ) -> Vec<&(std::string::String, CypherValue)> {
                let mut v: Vec<_> = m.iter().collect();
                v.sort_by(|p, q| p.0.cmp(&q.0));
                v
            }
            let (x, y) = (sorted(x), sorted(y));
            x.iter()
                .zip(&y)
                .map(|(p, q)| p.0.cmp(&q.0).then_with(|| order_cmp(&p.1, &q.1)))
                .find(|o| *o != Ordering::Equal)
                .unwrap_or_else(|| x.len().cmp(&y.len()))
        }
        (Null, Null) => Ordering::Equal,
        // Same-rank values without a defined order (temporal, spatial, bytes, paths) are
        // ordered by their representation, which is deterministic.
        _ => format!("{a:?}").cmp(&format!("{b:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use CypherValue::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn large_integers_are_not_equal_to_a_rounded_float() {
        // 2^53 + 1 is not representable; `as f64` would round it onto 2^53.
        let i = (1i64 << 53) + 1;
        assert_eq!(equals(&Int(i), &Float((1u64 << 53) as f64)), Some(false));
        assert_eq!(
            equals(&Int(1 << 53), &Float((1u64 << 53) as f64)),
            Some(true)
        );
        assert_eq!(
            order_cmp(&Int(i), &Float((1u64 << 53) as f64)),
            Ordering::Greater
        );
    }

    #[test]
    fn nan_is_unordered_for_comparison_and_largest_for_ordering() {
        assert_eq!(compare(&Float(f64::NAN), &Int(1)), Ok(None));
        assert_eq!(equals(&Float(f64::NAN), &Float(f64::NAN)), Some(false));
        assert_eq!(
            order_cmp(&Float(f64::NAN), &Int(i64::MAX)),
            Ordering::Greater
        );
        assert_eq!(order_cmp(&Float(f64::NAN), &Null), Ordering::Less);
    }

    #[test]
    fn list_equality_is_null_only_when_no_element_decides() {
        let l = |v: Vec<CypherValue>| List(v);
        assert_eq!(
            equals(&l(vec![Int(1), Null]), &l(vec![Int(2), Null])),
            Some(false)
        );
        assert_eq!(equals(&l(vec![Int(1), Null]), &l(vec![Int(1), Null])), None);
        assert_eq!(
            equals(&l(vec![Int(1)]), &l(vec![Int(1), Int(2)])),
            Some(false)
        );
    }
}
