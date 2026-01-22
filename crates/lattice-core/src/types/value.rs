//! Cypher query value types
//!
//! `CypherValue` represents all value types supported in Cypher queries,
//! including primitives, temporal types, spatial types, and graph references.
//!
//! # Memory Efficiency
//!
//! - Temporal types use components instead of `chrono` for WASM compatibility
//! - Uses standard Vec/String for simplicity (zero-copy optimization deferred)
//!
//! # Thread Safety
//!
//! - Native: Uses `Arc<str>` for strings, enabling parallel operations (Send + Sync)
//! - WASM: Uses `Rc<str>` for strings (no threading, lower overhead)
//!
//! # Future Optimization
//!
//! Zero-copy serialization via rkyv can be added once compact_str upgrades
//! to rkyv 0.8 (currently blocked by version mismatch).

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;

// =============================================================================
// SharedStr: Arc<str> on native (Send + Sync), Rc<str> on WASM (no overhead)
// =============================================================================

#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

/// Shared string type - Arc<str> on native for parallel ops, Rc<str> on WASM
#[cfg(not(target_arch = "wasm32"))]
pub type SharedStr = Arc<str>;
#[cfg(target_arch = "wasm32")]
pub type SharedStr = Rc<str>;

/// Custom serde support for SharedStr - serializes as String
mod shared_str_serde {
    use super::*;

    pub fn serialize<S>(value: &SharedStr, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(value)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SharedStr, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(SharedStr::from(s))
    }
}

/// A value in a Cypher query result or expression
///
/// # SSOT
///
/// This is THE authoritative representation of a Cypher value.
/// All conversions (to/from JSON, to/from storage) derive from this.
///
/// # Variants
///
/// - Primitives: Null, Bool, Int, Float, String, Bytes
/// - Temporal: Date, Time, DateTime, Duration
/// - Spatial: Point2D, Point3D
/// - Composite: List, Map
/// - Graph: NodeRef, RelationshipRef, Path
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum CypherValue {
    /// Null/missing value
    Null,

    /// Boolean value
    Bool(bool),

    /// 64-bit signed integer
    Int(i64),

    /// 64-bit floating point
    Float(f64),

    /// String value - uses SharedStr for O(1) cloning
    /// Arc<str> on native (enables parallel ops), Rc<str> on WASM
    #[serde(with = "shared_str_serde")]
    String(SharedStr),

    /// Binary data with inline storage for small arrays
    ///
    /// SmallVec stores up to 16 bytes inline.
    Bytes(SmallVec<[u8; 16]>),

    // --- Temporal Types (WASM-compatible, no chrono dependency) ---
    /// Date without time component
    Date { year: i32, month: u8, day: u8 },

    /// Time without date component
    Time {
        hour: u8,
        minute: u8,
        second: u8,
        nanos: u32,
    },

    /// Combined date and time
    ///
    /// Boxed to keep enum size reasonable.
    DateTime {
        date: Box<CypherValue>,
        time: Box<CypherValue>,
    },

    /// Duration between two instants
    Duration { months: i64, days: i64, nanos: i64 },

    // --- Spatial Types ---
    /// 2D point with spatial reference
    Point2D {
        x: f64,
        y: f64,
        /// Spatial Reference System Identifier (e.g., 4326 for WGS84)
        srid: u32,
    },

    /// 3D point with spatial reference
    Point3D {
        x: f64,
        y: f64,
        z: f64,
        /// Spatial Reference System Identifier
        srid: u32,
    },

    // --- Composite Types ---
    /// Ordered collection of values
    List(Vec<CypherValue>),

    /// Key-value pairs
    Map(Vec<(String, CypherValue)>),

    // --- Graph References (IDs, not full copies) ---
    /// Reference to a node by ID
    NodeRef(u64),

    /// Reference to a relationship by ID
    RelationshipRef(u64),

    /// Graph path as alternating node/relationship IDs
    ///
    /// Format: [node0, rel0, node1, rel1, node2, ...]
    /// SmallVec stores up to 8 elements inline (4 hops).
    Path(SmallVec<[u64; 8]>),
}

impl CypherValue {
    /// Create a null value
    pub fn null() -> Self {
        Self::Null
    }

    /// Create a boolean value
    pub fn bool(value: bool) -> Self {
        Self::Bool(value)
    }

    /// Create an integer value
    pub fn int(value: i64) -> Self {
        Self::Int(value)
    }

    /// Create a float value
    pub fn float(value: f64) -> Self {
        Self::Float(value)
    }

    /// Create a string value
    pub fn string(value: impl AsRef<str>) -> Self {
        Self::String(SharedStr::from(value.as_ref()))
    }

    /// Create a list from an iterator
    pub fn list(values: impl IntoIterator<Item = CypherValue>) -> Self {
        Self::List(values.into_iter().collect())
    }

    /// Create an empty map
    pub fn map() -> Self {
        Self::Map(Vec::new())
    }

    /// Create a map from key-value pairs
    pub fn map_from(pairs: impl IntoIterator<Item = (impl Into<String>, CypherValue)>) -> Self {
        Self::Map(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// Create a 2D point
    pub fn point2d(x: f64, y: f64, srid: u32) -> Self {
        Self::Point2D { x, y, srid }
    }

    /// Create a 3D point
    pub fn point3d(x: f64, y: f64, z: f64, srid: u32) -> Self {
        Self::Point3D { x, y, z, srid }
    }

    /// Create a node reference
    pub fn node_ref(id: u64) -> Self {
        Self::NodeRef(id)
    }

    /// Create a relationship reference
    pub fn relationship_ref(id: u64) -> Self {
        Self::RelationshipRef(id)
    }

    /// Create a date
    pub fn date(year: i32, month: u8, day: u8) -> Self {
        Self::Date { year, month, day }
    }

    /// Create a time
    pub fn time(hour: u8, minute: u8, second: u8, nanos: u32) -> Self {
        Self::Time {
            hour,
            minute,
            second,
            nanos,
        }
    }

    /// Create a duration
    pub fn duration(months: i64, days: i64, nanos: i64) -> Self {
        Self::Duration {
            months,
            days,
            nanos,
        }
    }

    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Check if value is truthy (non-null, non-false, non-empty)
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            Self::String(s) => !s.is_empty(),
            Self::Bytes(b) => !b.is_empty(),
            Self::List(l) => !l.is_empty(),
            Self::Map(m) => !m.is_empty(),
            _ => true, // Other types are truthy if they exist
        }
    }

    /// Get the type name for error messages
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "NULL",
            Self::Bool(_) => "BOOLEAN",
            Self::Int(_) => "INTEGER",
            Self::Float(_) => "FLOAT",
            Self::String(_) => "STRING",
            Self::Bytes(_) => "BYTES",
            Self::Date { .. } => "DATE",
            Self::Time { .. } => "TIME",
            Self::DateTime { .. } => "DATETIME",
            Self::Duration { .. } => "DURATION",
            Self::Point2D { .. } => "POINT",
            Self::Point3D { .. } => "POINT",
            Self::List(_) => "LIST",
            Self::Map(_) => "MAP",
            Self::NodeRef(_) => "NODE",
            Self::RelationshipRef(_) => "RELATIONSHIP",
            Self::Path(_) => "PATH",
        }
    }

    /// Try to convert to i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Try to convert to f64
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get as string reference
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

// Conversion from common types
impl From<bool> for CypherValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for CypherValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<i32> for CypherValue {
    fn from(value: i32) -> Self {
        Self::Int(value as i64)
    }
}

impl From<f64> for CypherValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<f32> for CypherValue {
    fn from(value: f32) -> Self {
        Self::Float(value as f64)
    }
}

impl From<String> for CypherValue {
    fn from(value: String) -> Self {
        Self::String(SharedStr::from(value))
    }
}

impl From<&str> for CypherValue {
    fn from(value: &str) -> Self {
        Self::String(SharedStr::from(value))
    }
}

impl From<SharedStr> for CypherValue {
    fn from(value: SharedStr) -> Self {
        Self::String(value)
    }
}

impl<T: Into<CypherValue>> From<Option<T>> for CypherValue {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => v.into(),
            None => Self::Null,
        }
    }
}

impl<T: Into<CypherValue>> From<Vec<T>> for CypherValue {
    fn from(value: Vec<T>) -> Self {
        Self::List(value.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_cypher_value_creation() {
        assert!(CypherValue::null().is_null());
        assert_eq!(CypherValue::bool(true), CypherValue::Bool(true));
        assert_eq!(CypherValue::int(42), CypherValue::Int(42));
        assert_eq!(CypherValue::float(3.14), CypherValue::Float(3.14));
        assert_eq!(
            CypherValue::string("hello"),
            CypherValue::String(SharedStr::from("hello"))
        );
    }

    #[test]
    fn test_cypher_value_type_names() {
        assert_eq!(CypherValue::Null.type_name(), "NULL");
        assert_eq!(CypherValue::Bool(true).type_name(), "BOOLEAN");
        assert_eq!(CypherValue::Int(42).type_name(), "INTEGER");
        assert_eq!(CypherValue::Float(3.14).type_name(), "FLOAT");
        assert_eq!(CypherValue::string("test").type_name(), "STRING");
        assert_eq!(CypherValue::list([]).type_name(), "LIST");
        assert_eq!(CypherValue::map().type_name(), "MAP");
        assert_eq!(CypherValue::node_ref(1).type_name(), "NODE");
        assert_eq!(CypherValue::relationship_ref(1).type_name(), "RELATIONSHIP");
    }

    #[test]
    fn test_cypher_value_truthy() {
        assert!(!CypherValue::Null.is_truthy());
        assert!(!CypherValue::Bool(false).is_truthy());
        assert!(CypherValue::Bool(true).is_truthy());
        assert!(!CypherValue::Int(0).is_truthy());
        assert!(CypherValue::Int(1).is_truthy());
        assert!(!CypherValue::string("").is_truthy());
        assert!(CypherValue::string("x").is_truthy());
    }

    #[test]
    fn test_cypher_value_conversions() {
        assert_eq!(CypherValue::int(42).as_int(), Some(42));
        assert_eq!(CypherValue::float(3.14).as_float(), Some(3.14));
        assert_eq!(CypherValue::int(42).as_float(), Some(42.0));
        assert_eq!(CypherValue::string("hello").as_str(), Some("hello"));
        assert_eq!(CypherValue::Bool(true).as_bool(), Some(true));
    }

    #[test]
    fn test_cypher_value_from_traits() {
        let v: CypherValue = 42i64.into();
        assert_eq!(v, CypherValue::Int(42));

        let v: CypherValue = 3.14f64.into();
        assert_eq!(v, CypherValue::Float(3.14));

        let v: CypherValue = "hello".into();
        assert_eq!(v, CypherValue::string("hello"));

        let v: CypherValue = true.into();
        assert_eq!(v, CypherValue::Bool(true));

        let v: CypherValue = None::<i64>.into();
        assert_eq!(v, CypherValue::Null);

        let v: CypherValue = Some(42i64).into();
        assert_eq!(v, CypherValue::Int(42));
    }

    #[test]
    fn test_cypher_value_spatial() {
        let p2d = CypherValue::point2d(40.7128, -74.0060, 4326);
        assert_eq!(p2d.type_name(), "POINT");

        let p3d = CypherValue::point3d(40.7128, -74.0060, 10.0, 4326);
        assert_eq!(p3d.type_name(), "POINT");
    }

    #[test]
    fn test_cypher_value_temporal() {
        let date = CypherValue::date(2024, 6, 15);
        assert_eq!(date.type_name(), "DATE");

        let time = CypherValue::time(14, 30, 0, 0);
        assert_eq!(time.type_name(), "TIME");

        let duration = CypherValue::duration(0, 7, 0);
        assert_eq!(duration.type_name(), "DURATION");
    }

    #[test]
    fn test_cypher_value_list_and_map() {
        let list = CypherValue::list([
            CypherValue::int(1),
            CypherValue::int(2),
            CypherValue::int(3),
        ]);
        assert_eq!(list.type_name(), "LIST");

        let map = CypherValue::map_from([
            ("name", CypherValue::string("Alice")),
            ("age", CypherValue::int(30)),
        ]);
        assert_eq!(map.type_name(), "MAP");
    }
}
