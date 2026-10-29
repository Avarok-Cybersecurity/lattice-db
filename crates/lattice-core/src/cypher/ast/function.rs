//! The function catalogue: every function name the engine evaluates, with its arity.
//!
//! This is the single source of truth for function names. The planner resolves a parsed
//! `FunctionCall` against it once (aggregates become aggregation slots, scalars become
//! `Expr::Scalar`), so the executor never compares function-name strings per row.

use serde::{Deserialize, Serialize};

/// Aggregate functions (they collapse the rows of a group into one value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregateKind {
    Count,
    Collect,
    Min,
    Max,
    Sum,
    Avg,
}

impl AggregateKind {
    const ALL: [AggregateKind; 6] = [
        Self::Count,
        Self::Collect,
        Self::Min,
        Self::Max,
        Self::Sum,
        Self::Avg,
    ];

    /// The canonical (openCypher) spelling
    pub fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Collect => "collect",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
            Self::Avg => "avg",
        }
    }

    /// Look a name up, case-insensitively (openCypher function names are case-insensitive)
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|k| k.name().eq_ignore_ascii_case(name))
    }
}

/// Scalar functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarFn {
    // Graph
    Labels,
    Type,
    Id,
    Keys,
    Properties,
    StartNode,
    EndNode,
    // Predicates / null handling
    Exists,
    Coalesce,
    // Collections and strings
    Size,
    Length,
    Head,
    Last,
    Tail,
    Reverse,
    Range,
    // Conversions
    ToString,
    ToInteger,
    ToFloat,
    ToBoolean,
    // Strings
    ToLower,
    ToUpper,
    Trim,
    LTrim,
    RTrim,
    Replace,
    Substring,
    Split,
    // Numbers
    Abs,
    Ceil,
    Floor,
    Round,
    Sqrt,
    Sign,
}

/// Maximum argument count meaning "any number" (coalesce)
const VARIADIC: usize = usize::MAX;

impl ScalarFn {
    const ALL: [ScalarFn; 34] = [
        Self::Labels,
        Self::Type,
        Self::Id,
        Self::Keys,
        Self::Properties,
        Self::StartNode,
        Self::EndNode,
        Self::Exists,
        Self::Coalesce,
        Self::Size,
        Self::Length,
        Self::Head,
        Self::Last,
        Self::Tail,
        Self::Reverse,
        Self::Range,
        Self::ToString,
        Self::ToInteger,
        Self::ToFloat,
        Self::ToBoolean,
        Self::ToLower,
        Self::ToUpper,
        Self::Trim,
        Self::LTrim,
        Self::RTrim,
        Self::Replace,
        Self::Substring,
        Self::Split,
        Self::Abs,
        Self::Ceil,
        Self::Floor,
        Self::Round,
        Self::Sqrt,
        Self::Sign,
    ];

    /// The canonical (openCypher) spelling
    pub fn name(self) -> &'static str {
        match self {
            Self::Labels => "labels",
            Self::Type => "type",
            Self::Id => "id",
            Self::Keys => "keys",
            Self::Properties => "properties",
            Self::StartNode => "startNode",
            Self::EndNode => "endNode",
            Self::Exists => "exists",
            Self::Coalesce => "coalesce",
            Self::Size => "size",
            Self::Length => "length",
            Self::Head => "head",
            Self::Last => "last",
            Self::Tail => "tail",
            Self::Reverse => "reverse",
            Self::Range => "range",
            Self::ToString => "toString",
            Self::ToInteger => "toInteger",
            Self::ToFloat => "toFloat",
            Self::ToBoolean => "toBoolean",
            Self::ToLower => "toLower",
            Self::ToUpper => "toUpper",
            Self::Trim => "trim",
            Self::LTrim => "ltrim",
            Self::RTrim => "rtrim",
            Self::Replace => "replace",
            Self::Substring => "substring",
            Self::Split => "split",
            Self::Abs => "abs",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::Round => "round",
            Self::Sqrt => "sqrt",
            Self::Sign => "sign",
        }
    }

    /// Inclusive (min, max) argument counts
    pub fn arity(self) -> (usize, usize) {
        match self {
            Self::Coalesce => (1, VARIADIC),
            Self::Range | Self::Substring => (2, 3),
            Self::Replace => (3, 3),
            Self::Split => (2, 2),
            _ => (1, 1),
        }
    }

    /// Look a name up, case-insensitively
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.name().eq_ignore_ascii_case(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn every_catalogued_name_round_trips_case_insensitively() {
        for f in ScalarFn::ALL {
            assert_eq!(ScalarFn::from_name(f.name()), Some(f));
            assert_eq!(ScalarFn::from_name(&f.name().to_uppercase()), Some(f));
            assert_eq!(
                AggregateKind::from_name(f.name()),
                None,
                "{f:?} is not an aggregate"
            );
        }
        for k in AggregateKind::ALL {
            assert_eq!(AggregateKind::from_name(&k.name().to_uppercase()), Some(k));
            assert_eq!(ScalarFn::from_name(k.name()), None, "{k:?} is not a scalar");
        }
        assert_eq!(ScalarFn::from_name("frobnicate"), None);
    }
}
