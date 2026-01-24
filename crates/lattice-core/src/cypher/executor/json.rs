//! JSON conversion utilities for CypherValue

use super::QueryExecutor;
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;
use crate::types::SharedStr;

impl QueryExecutor {
    /// Convert JSON bytes to CypherValue
    /// Optimized with fast paths for simple types to avoid full JSON parsing
    #[inline]
    pub(crate) fn json_bytes_to_cypher_value(&self, bytes: &[u8]) -> CypherResult<CypherValue> {
        // Fast path for common simple types - avoids serde_json overhead
        if bytes.is_empty() {
            return Ok(CypherValue::Null);
        }

        match bytes[0] {
            // Null
            b'n' if bytes == b"null" => return Ok(CypherValue::Null),

            // Boolean
            b't' if bytes == b"true" => return Ok(CypherValue::Bool(true)),
            b'f' if bytes == b"false" => return Ok(CypherValue::Bool(false)),

            // Integer (fast path - direct parsing without iterator allocation)
            b'0'..=b'9' => {
                // Fast path for positive integers: parse directly
                if let Some(val) = self.fast_parse_positive_int(bytes) {
                    return Ok(CypherValue::Int(val));
                }
                // Fall through to serde_json for floats or complex numbers
            }
            b'-' if bytes.len() > 1 => {
                // Fast path for negative integers
                if let Some(val) = self.fast_parse_negative_int(bytes) {
                    return Ok(CypherValue::Int(val));
                }
                // Fall through to serde_json for floats or complex numbers
            }

            // String (fast path for simple strings without escapes)
            b'"' if bytes.len() >= 2 && bytes[bytes.len() - 1] == b'"' => {
                // Check if it's a simple string (no escape sequences)
                let inner = &bytes[1..bytes.len() - 1];
                if !inner.contains(&b'\\') {
                    if let Ok(s) = std::str::from_utf8(inner) {
                        return Ok(CypherValue::String(SharedStr::from(s)));
                    }
                }
                // Fall through to serde_json for strings with escapes
            }

            _ => {}
        }

        // Fall back to serde_json for complex types (arrays, objects, escaped strings)
        let json: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| CypherError::Internal {
                message: e.to_string(),
            })?;

        self.json_to_cypher_value(&json)
    }

    /// Fast parse a positive integer from bytes without any allocation
    #[inline]
    pub(crate) fn fast_parse_positive_int(&self, bytes: &[u8]) -> Option<i64> {
        let mut result: i64 = 0;
        for &b in bytes {
            if b.is_ascii_digit() {
                result = result.checked_mul(10)?.checked_add((b - b'0') as i64)?;
            } else {
                // Not a simple integer (float, exponent, etc.)
                return None;
            }
        }
        Some(result)
    }

    /// Fast parse a negative integer from bytes without any allocation
    #[inline]
    pub(crate) fn fast_parse_negative_int(&self, bytes: &[u8]) -> Option<i64> {
        if bytes.len() < 2 || bytes[0] != b'-' {
            return None;
        }
        let mut result: i64 = 0;
        for &b in &bytes[1..] {
            if b.is_ascii_digit() {
                result = result.checked_mul(10)?.checked_sub((b - b'0') as i64)?;
            } else {
                // Not a simple integer (float, exponent, etc.)
                return None;
            }
        }
        Some(result)
    }

    /// Convert JSON value to CypherValue
    pub(crate) fn json_to_cypher_value(
        &self,
        json: &serde_json::Value,
    ) -> CypherResult<CypherValue> {
        match json {
            serde_json::Value::Null => Ok(CypherValue::Null),
            serde_json::Value::Bool(b) => Ok(CypherValue::Bool(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(CypherValue::Int(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(CypherValue::Float(f))
                } else {
                    Ok(CypherValue::Null)
                }
            }
            serde_json::Value::String(s) => Ok(CypherValue::String(SharedStr::from(s.as_str()))),
            serde_json::Value::Array(arr) => {
                let values: CypherResult<Vec<_>> =
                    arr.iter().map(|v| self.json_to_cypher_value(v)).collect();
                Ok(CypherValue::list(values?))
            }
            serde_json::Value::Object(obj) => {
                let entries: CypherResult<Vec<_>> = obj
                    .iter()
                    .map(|(k, v)| Ok((k.as_str(), self.json_to_cypher_value(v)?)))
                    .collect();
                Ok(CypherValue::map_from(entries?))
            }
        }
    }

    /// Convert CypherValue to JSON bytes
    pub(crate) fn cypher_value_to_json_bytes(&self, value: &CypherValue) -> CypherResult<Vec<u8>> {
        let json = self.cypher_value_to_json(value)?;
        serde_json::to_vec(&json).map_err(|e| CypherError::Internal {
            message: e.to_string(),
        })
    }

    /// Convert CypherValue to JSON
    pub(crate) fn cypher_value_to_json(
        &self,
        value: &CypherValue,
    ) -> CypherResult<serde_json::Value> {
        match value {
            CypherValue::Null => Ok(serde_json::Value::Null),
            CypherValue::Bool(b) => Ok(serde_json::Value::Bool(*b)),
            CypherValue::Int(i) => Ok(serde_json::Value::Number((*i).into())),
            CypherValue::Float(f) => Ok(serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0)),
            )),
            CypherValue::String(s) => Ok(serde_json::Value::String((*s).to_string())),
            CypherValue::List(items) => {
                let json_items: CypherResult<Vec<_>> =
                    items.iter().map(|v| self.cypher_value_to_json(v)).collect();
                Ok(serde_json::Value::Array(json_items?))
            }
            CypherValue::Map(entries) => {
                let mut obj = serde_json::Map::new();
                for (k, v) in entries.iter() {
                    obj.insert(k.to_string(), self.cypher_value_to_json(v)?);
                }
                Ok(serde_json::Value::Object(obj))
            }
            _ => Ok(serde_json::Value::Null), // Simplify complex types to null
        }
    }
}
