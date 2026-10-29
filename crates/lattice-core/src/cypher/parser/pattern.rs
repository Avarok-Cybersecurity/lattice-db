//! Pattern parsing: node and relationship patterns, path lengths.

use super::{CypherParser, Rule, MAX_PATH_LENGTH};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use pest::iterators::Pair;

impl CypherParser {
    /// Parse pattern list
    pub(super) fn parse_pattern_list(&self, pair: Pair<Rule>) -> CypherResult<Vec<Pattern>> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern {
                patterns.push(self.parse_pattern(inner)?);
            }
        }

        Ok(patterns)
    }

    /// Parse a single pattern
    pub(super) fn parse_pattern(&self, pair: Pair<Rule>) -> CypherResult<Pattern> {
        let mut elements = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::node_pattern => {
                    elements.push(PatternElement::Node(self.parse_node_pattern(inner)?));
                }
                Rule::relationship_pattern => {
                    elements.push(PatternElement::Relationship(
                        self.parse_relationship_pattern(inner)?,
                    ));
                }
                _ => {}
            }
        }

        Ok(Pattern { elements })
    }

    /// Parse node pattern
    pub(super) fn parse_node_pattern(&self, pair: Pair<Rule>) -> CypherResult<NodePattern> {
        let mut variable = None;
        let mut labels = Vec::new();
        let mut properties = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier => {
                    variable = Some(String::from(inner.as_str()));
                }
                Rule::label_list => {
                    for label_inner in inner.into_inner() {
                        if label_inner.as_rule() == Rule::identifier {
                            labels.push(String::from(label_inner.as_str()));
                        }
                    }
                }
                Rule::properties => {
                    properties = Some(self.parse_map_literal(inner)?);
                }
                _ => {}
            }
        }

        Ok(NodePattern {
            variable,
            labels,
            properties,
        })
    }

    /// Parse relationship pattern
    pub(super) fn parse_relationship_pattern(&self, pair: Pair<Rule>) -> CypherResult<RelPattern> {
        let mut variable = None;
        let mut rel_types = Vec::new();
        let mut properties = None;
        let mut length = None;

        let mut has_left_arrow = false;
        let mut has_right_arrow = false;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::left_arrow => {
                    has_left_arrow = true;
                }
                Rule::right_arrow => {
                    has_right_arrow = true;
                }
                Rule::rel_detail => {
                    for detail in inner.into_inner() {
                        match detail.as_rule() {
                            Rule::identifier => {
                                variable = Some(String::from(detail.as_str()));
                            }
                            Rule::rel_type_list => {
                                for type_inner in detail.into_inner() {
                                    if type_inner.as_rule() == Rule::identifier {
                                        rel_types.push(String::from(type_inner.as_str()));
                                    }
                                }
                            }
                            Rule::path_length => {
                                length = Some(self.parse_path_length(detail)?);
                            }
                            Rule::properties => {
                                properties = Some(self.parse_map_literal(detail)?);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        let direction = match (has_left_arrow, has_right_arrow) {
            (true, false) => Direction::Incoming,
            (false, true) => Direction::Outgoing,
            _ => Direction::Both,
        };

        Ok(RelPattern {
            variable,
            rel_types,
            direction,
            properties,
            length,
        })
    }

    /// Parse path length
    pub(super) fn parse_path_length(&self, pair: Pair<Rule>) -> CypherResult<PathLength> {
        let mut min = None;
        let mut max = None;

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::path_range {
                let range_str = inner.as_str();
                if range_str.contains("..") {
                    let parts: Vec<&str> = range_str.split("..").collect();
                    // Validate exactly 2 parts for range syntax
                    if parts.len() != 2 {
                        return Err(CypherError::syntax(
                            0,
                            0,
                            format!("Invalid path range syntax: '{}'", range_str),
                        ));
                    }
                    if !parts[0].is_empty() {
                        min = Some(parts[0].parse().map_err(|_| {
                            CypherError::syntax(
                                0,
                                0,
                                format!("Invalid path minimum: '{}'", parts[0]),
                            )
                        })?);
                    }
                    if !parts[1].is_empty() {
                        max = Some(parts[1].parse().map_err(|_| {
                            CypherError::syntax(
                                0,
                                0,
                                format!("Invalid path maximum: '{}'", parts[1]),
                            )
                        })?);
                    }
                } else {
                    let n: u32 = range_str.parse().map_err(|_| {
                        CypherError::syntax(0, 0, format!("Invalid path length: '{}'", range_str))
                    })?;
                    min = Some(n);
                    max = Some(n);
                }
            }
        }

        // Apply security limits (DoS protection)
        // Default unbounded paths to MAX_PATH_LENGTH instead of u32::MAX
        let effective_max = max.unwrap_or(MAX_PATH_LENGTH);
        if effective_max > MAX_PATH_LENGTH {
            return Err(CypherError::syntax(
                0,
                0,
                format!(
                    "Path length {} exceeds maximum allowed depth of {}",
                    effective_max, MAX_PATH_LENGTH
                ),
            ));
        }

        // Validate min doesn't exceed max
        if let (Some(min_val), Some(max_val)) = (min, max) {
            if min_val > max_val {
                return Err(CypherError::syntax(
                    0,
                    0,
                    format!(
                        "Invalid path range: minimum {} exceeds maximum {}",
                        min_val, max_val
                    ),
                ));
            }
        }

        Ok(PathLength { min, max })
    }
}
