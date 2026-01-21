//! Cypher query error types
//!
//! All errors are explicit with detailed context for debugging.

use thiserror::Error;

/// Cypher query execution errors
#[derive(Debug, Error)]
pub enum CypherError {
    // --- Parser Errors ---
    #[error("Syntax error at line {line}, column {column}: {message}")]
    SyntaxError {
        line: usize,
        column: usize,
        message: String,
    },

    #[error("Unexpected token: expected {expected}, found '{found}'")]
    UnexpectedToken { expected: String, found: String },

    #[error("Unexpected end of input")]
    UnexpectedEof,

    // --- Semantic Errors ---
    #[error("Unknown variable: '{name}'")]
    UnknownVariable { name: String },

    #[error("Unknown label: '{label}'")]
    UnknownLabel { label: String },

    #[error("Unknown relationship type: '{rel_type}'")]
    UnknownRelationType { rel_type: String },

    #[error("Unknown property: '{property}' on variable '{variable}'")]
    UnknownProperty { variable: String, property: String },

    #[error("Type mismatch: expected {expected}, found {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Cannot compare {left_type} with {right_type}")]
    IncomparableTypes {
        left_type: String,
        right_type: String,
    },

    #[error("Invalid operation: {operation} on {value_type}")]
    InvalidOperation {
        operation: String,
        value_type: String,
    },

    // --- Execution Errors ---
    #[error("Node not found: {id}")]
    NodeNotFound { id: u64 },

    #[error("Relationship not found: {id}")]
    RelationshipNotFound { id: u64 },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Null property access: '{property}' on null value")]
    NullPropertyAccess { property: String },

    // --- Constraint Errors ---
    #[error("Query too complex: {message}")]
    QueryTooComplex { message: String },

    #[error("Result set too large: limit is {limit}, got {actual}")]
    ResultTooLarge { limit: usize, actual: usize },

    #[error("Maximum depth exceeded: limit is {limit}")]
    MaxDepthExceeded { limit: usize },

    // --- Feature Errors ---
    #[error("Unsupported feature: {feature}")]
    UnsupportedFeature { feature: String },

    #[error("Unsupported clause: {clause}")]
    UnsupportedClause { clause: String },

    // --- Internal Errors ---
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl CypherError {
    /// Create a syntax error with location
    pub fn syntax(line: usize, column: usize, message: impl Into<String>) -> Self {
        Self::SyntaxError {
            line,
            column,
            message: message.into(),
        }
    }

    /// Create an unexpected token error
    pub fn unexpected_token(expected: impl Into<String>, found: impl Into<String>) -> Self {
        Self::UnexpectedToken {
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Create an unknown variable error
    pub fn unknown_variable(name: impl Into<String>) -> Self {
        Self::UnknownVariable { name: name.into() }
    }

    /// Create a type mismatch error
    pub fn type_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::TypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create an unsupported feature error
    pub fn unsupported(feature: impl Into<String>) -> Self {
        Self::UnsupportedFeature {
            feature: feature.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

/// Convenience type alias for Cypher results
pub type CypherResult<T> = Result<T, CypherError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_error_messages() {
        let err = CypherError::syntax(1, 5, "unexpected character");
        assert!(err.to_string().contains("line 1"));
        assert!(err.to_string().contains("column 5"));

        let err = CypherError::unknown_variable("x");
        assert!(err.to_string().contains("'x'"));

        let err = CypherError::type_mismatch("INTEGER", "STRING");
        assert!(err.to_string().contains("INTEGER"));
        assert!(err.to_string().contains("STRING"));
    }

    #[test]
    fn test_error_constructors() {
        let _ = CypherError::unexpected_token("MATCH", "CREATE");
        let _ = CypherError::unsupported("CALL procedures");
        let _ = CypherError::internal("unexpected state");
    }
}
