//! Expression evaluation for Cypher queries

use crate::cypher::ast::{BinaryOp, Expr, UnaryOp};
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::point::Point;
use crate::types::value::CypherValue;
use crate::types::SharedStr;
use super::{ExecutionContext, QueryExecutor};

impl QueryExecutor {
    /// Evaluate a predicate expression
    pub(crate) fn evaluate_predicate(
        &self,
        predicate: &Expr,
        row: &[CypherValue],
        ctx: &ExecutionContext,
    ) -> CypherResult<bool> {
        let value = self.evaluate_expr(predicate, row, ctx)?;
        Ok(value.is_truthy())
    }

    /// Evaluate an expression
    pub(crate) fn evaluate_expr(
        &self,
        expr: &Expr,
        row: &[CypherValue],
        ctx: &ExecutionContext,
    ) -> CypherResult<CypherValue> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),

            Expr::Variable(name) => {
                // Check parameters first
                if let Some(value) = ctx.parameters.get(name.as_str()) {
                    return Ok(value.clone());
                }

                // For now, assume first column is the variable
                // TODO: Proper variable binding
                row.first()
                    .cloned()
                    .ok_or_else(|| CypherError::unknown_variable(name.as_str()))
            }

            Expr::Property { expr, property } => {
                let base = self.evaluate_expr(expr, row, ctx)?;

                match base {
                    CypherValue::NodeRef(id) => {
                        // Check property cache first to avoid JSON deserialization
                        if let Some(cached) = ctx.get_property_cached(id, property) {
                            return Ok(cached);
                        }

                        // Get the point using point cache and extract property
                        if let Some(point) = ctx.get_point_cached(id) {
                            let value = self.get_point_property(&point, property)?;
                            // Cache the extracted property value
                            ctx.cache_property(id, property, value.clone());
                            Ok(value)
                        } else {
                            Ok(CypherValue::Null)
                        }
                    }
                    CypherValue::Map(entries) => {
                        // Find property in map
                        entries
                            .iter()
                            .find(|(k, _)| k.as_str() == property.as_str())
                            .map(|(_, v)| v.clone())
                            .ok_or_else(|| CypherError::UnknownProperty {
                                variable: "map".to_string(),
                                property: property.to_string(),
                            })
                    }
                    _ => Err(CypherError::InvalidOperation {
                        operation: "property access".to_string(),
                        value_type: base.type_name().to_string(),
                    }),
                }
            }

            Expr::Parameter(name) => ctx
                .parameters
                .get(name.as_str())
                .cloned()
                .ok_or_else(|| CypherError::unknown_variable(name.as_str())),

            Expr::BinaryOp { left, op, right } => {
                // Short-circuit evaluation for AND and OR
                match op {
                    BinaryOp::And => {
                        let left_val = self.evaluate_expr(left, row, ctx)?;
                        if !left_val.is_truthy() {
                            return Ok(CypherValue::Bool(false));
                        }
                        let right_val = self.evaluate_expr(right, row, ctx)?;
                        Ok(CypherValue::Bool(right_val.is_truthy()))
                    }
                    BinaryOp::Or => {
                        let left_val = self.evaluate_expr(left, row, ctx)?;
                        if left_val.is_truthy() {
                            return Ok(CypherValue::Bool(true));
                        }
                        let right_val = self.evaluate_expr(right, row, ctx)?;
                        Ok(CypherValue::Bool(right_val.is_truthy()))
                    }
                    _ => {
                        // Non-logical operators: evaluate both sides
                        let left_val = self.evaluate_expr(left, row, ctx)?;
                        let right_val = self.evaluate_expr(right, row, ctx)?;
                        self.evaluate_binary_op(*op, &left_val, &right_val)
                    }
                }
            }

            Expr::UnaryOp { op, expr } => {
                let val = self.evaluate_expr(expr, row, ctx)?;
                self.evaluate_unary_op(*op, &val)
            }

            Expr::IsNull { expr, negated } => {
                let val = self.evaluate_expr(expr, row, ctx)?;
                let is_null = val.is_null();
                Ok(CypherValue::Bool(if *negated { !is_null } else { is_null }))
            }

            Expr::List(items) => {
                let values: CypherResult<Vec<_>> = items
                    .iter()
                    .map(|item| self.evaluate_expr(item, row, ctx))
                    .collect();
                Ok(CypherValue::list(values?))
            }

            Expr::Star => {
                // Return all values as a list
                Ok(CypherValue::list(row.to_vec()))
            }

            _ => Err(CypherError::unsupported("Complex expression evaluation")),
        }
    }

    /// Evaluate a binary operation
    pub(crate) fn evaluate_binary_op(
        &self,
        op: BinaryOp,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        match op {
            // Arithmetic
            BinaryOp::Add => self.add_values(left, right),
            BinaryOp::Sub => self.sub_values(left, right),
            BinaryOp::Mul => self.mul_values(left, right),
            BinaryOp::Div => self.div_values(left, right),
            BinaryOp::Mod => self.mod_values(left, right),

            // Comparison
            BinaryOp::Eq => Ok(CypherValue::Bool(left == right)),
            BinaryOp::Neq => Ok(CypherValue::Bool(left != right)),
            BinaryOp::Lt => Ok(CypherValue::Bool(self.compare_values(left, right).is_lt())),
            BinaryOp::Lte => Ok(CypherValue::Bool(self.compare_values(left, right).is_le())),
            BinaryOp::Gt => Ok(CypherValue::Bool(self.compare_values(left, right).is_gt())),
            BinaryOp::Gte => Ok(CypherValue::Bool(self.compare_values(left, right).is_ge())),

            // Logical
            BinaryOp::And => Ok(CypherValue::Bool(left.is_truthy() && right.is_truthy())),
            BinaryOp::Or => Ok(CypherValue::Bool(left.is_truthy() || right.is_truthy())),
            BinaryOp::Xor => Ok(CypherValue::Bool(left.is_truthy() ^ right.is_truthy())),

            // String concatenation - creates new Rc<str>
            BinaryOp::Concat => {
                let left_str = left.as_str().unwrap_or("");
                let right_str = right.as_str().unwrap_or("");
                let result = format!("{}{}", left_str, right_str);
                Ok(CypherValue::String(SharedStr::from(result)))
            }

            _ => Err(CypherError::unsupported("Unsupported binary operation")),
        }
    }

    /// Evaluate a unary operation
    pub(crate) fn evaluate_unary_op(&self, op: UnaryOp, value: &CypherValue) -> CypherResult<CypherValue> {
        match op {
            UnaryOp::Not => Ok(CypherValue::Bool(!value.is_truthy())),
            UnaryOp::Neg => match value {
                CypherValue::Int(i) => Ok(CypherValue::Int(-i)),
                CypherValue::Float(f) => Ok(CypherValue::Float(-f)),
                _ => Err(CypherError::InvalidOperation {
                    operation: "negation".to_string(),
                    value_type: value.type_name().to_string(),
                }),
            },
            UnaryOp::Pos => Ok(value.clone()),
        }
    }

    /// Add two values
    pub(crate) fn add_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => Ok(CypherValue::Int(a + b)),
            (CypherValue::Float(a), CypherValue::Float(b)) => Ok(CypherValue::Float(a + b)),
            (CypherValue::Int(a), CypherValue::Float(b)) => Ok(CypherValue::Float(*a as f64 + b)),
            (CypherValue::Float(a), CypherValue::Int(b)) => Ok(CypherValue::Float(a + *b as f64)),
            (CypherValue::String(a), CypherValue::String(b)) => {
                let result = format!("{}{}", a, b);
                Ok(CypherValue::String(SharedStr::from(result)))
            }
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Subtract two values
    pub(crate) fn sub_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => Ok(CypherValue::Int(a - b)),
            (CypherValue::Float(a), CypherValue::Float(b)) => Ok(CypherValue::Float(a - b)),
            (CypherValue::Int(a), CypherValue::Float(b)) => Ok(CypherValue::Float(*a as f64 - b)),
            (CypherValue::Float(a), CypherValue::Int(b)) => Ok(CypherValue::Float(a - *b as f64)),
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Multiply two values
    pub(crate) fn mul_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => Ok(CypherValue::Int(a * b)),
            (CypherValue::Float(a), CypherValue::Float(b)) => Ok(CypherValue::Float(a * b)),
            (CypherValue::Int(a), CypherValue::Float(b)) => Ok(CypherValue::Float(*a as f64 * b)),
            (CypherValue::Float(a), CypherValue::Int(b)) => Ok(CypherValue::Float(a * *b as f64)),
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Divide two values
    pub(crate) fn div_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => {
                if *b == 0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Int(a / b))
                }
            }
            (CypherValue::Float(a), CypherValue::Float(b)) => {
                if *b == 0.0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(a / b))
                }
            }
            (CypherValue::Int(a), CypherValue::Float(b)) => {
                if *b == 0.0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(*a as f64 / b))
                }
            }
            (CypherValue::Float(a), CypherValue::Int(b)) => {
                if *b == 0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(a / *b as f64))
                }
            }
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Modulo two values
    pub(crate) fn mod_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => {
                if *b == 0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Int(a % b))
                }
            }
            (CypherValue::Float(a), CypherValue::Float(b)) => {
                if *b == 0.0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(a % b))
                }
            }
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Compare two values
    pub(crate) fn compare_values(&self, left: &CypherValue, right: &CypherValue) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        match (left, right) {
            (CypherValue::Null, CypherValue::Null) => Ordering::Equal,
            (CypherValue::Null, _) => Ordering::Less,
            (_, CypherValue::Null) => Ordering::Greater,

            (CypherValue::Bool(a), CypherValue::Bool(b)) => a.cmp(b),
            (CypherValue::Int(a), CypherValue::Int(b)) => a.cmp(b),
            (CypherValue::Float(a), CypherValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (CypherValue::Int(a), CypherValue::Float(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (CypherValue::Float(a), CypherValue::Int(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (CypherValue::String(a), CypherValue::String(b)) => a.cmp(b),

            _ => Ordering::Equal, // Default for incompatible types
        }
    }

    /// Check if a point has a specific label
    #[allow(dead_code)]
    pub(crate) fn point_has_label(&self, point: &Point, label: &str) -> bool {
        if let Some(labels_bytes) = point.payload.get("_labels") {
            if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                return labels.iter().any(|l| l == label);
            }
        }
        false
    }

    /// Get a property from a point
    pub(crate) fn get_point_property(&self, point: &Point, property: &str) -> CypherResult<CypherValue> {
        // Check for special properties
        match property {
            "id" => return Ok(CypherValue::Int(point.id as i64)),
            "_labels" => {
                if let Some(labels_bytes) = point.payload.get("_labels") {
                    if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                        return Ok(CypherValue::list(
                            labels
                                .into_iter()
                                .map(CypherValue::from)
                                .collect::<Vec<_>>(),
                        ));
                    }
                }
                return Ok(CypherValue::list(vec![]));
            }
            _ => {}
        }

        // Get from payload
        if let Some(value_bytes) = point.payload.get(property) {
            self.json_bytes_to_cypher_value(value_bytes)
        } else {
            Ok(CypherValue::Null)
        }
    }
}
