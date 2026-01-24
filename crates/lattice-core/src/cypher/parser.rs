//! Cypher query parser using pest grammar
//!
//! Converts Cypher query strings into AST structures.

use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;
// compact_str removed - using standard String for simplicity
use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
// smallvec removed - AST now uses Vec for simplicity

#[derive(Parser)]
#[grammar = "cypher/cypher.pest"]
struct CypherPestParser;

/// Cypher query parser
///
/// Parses Cypher query strings into AST structures for further processing.
pub struct CypherParser;

impl CypherParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self
    }

    /// Parse a Cypher statement
    pub fn parse(&self, input: &str) -> CypherResult<Statement> {
        let pairs = CypherPestParser::parse(Rule::statement, input)
            .map_err(|e| self.pest_error_to_cypher(e))?;

        self.parse_statement(pairs)
    }

    /// Parse a statement from pest pairs
    fn parse_statement(&self, pairs: Pairs<Rule>) -> CypherResult<Statement> {
        for pair in pairs {
            match pair.as_rule() {
                Rule::query => return Ok(Statement::Query(self.parse_query(pair)?)),
                Rule::create_statement => {
                    return Ok(Statement::Create(self.parse_create_statement(pair)?))
                }
                Rule::delete_statement => {
                    return Ok(Statement::Delete(self.parse_delete_statement(pair)?))
                }
                Rule::EOI => continue,
                _ => continue,
            }
        }
        Err(CypherError::UnexpectedEof)
    }

    /// Parse a query
    fn parse_query(&self, pair: Pair<Rule>) -> CypherResult<Query> {
        let mut match_clause = None;
        let mut optional_matches = Vec::new();
        let mut where_clause = None;
        let mut return_clause = None;
        let mut order_by = None;
        let mut skip = None;
        let mut limit = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::match_clause => {
                    match_clause = Some(self.parse_match_clause(inner)?);
                }
                Rule::optional_match => {
                    optional_matches.push(self.parse_optional_match(inner)?);
                }
                Rule::where_clause => {
                    where_clause = Some(self.parse_where_clause(inner)?);
                }
                Rule::return_clause => {
                    return_clause = Some(self.parse_return_clause(inner)?);
                }
                Rule::order_by_clause => {
                    order_by = Some(self.parse_order_by_clause(inner)?);
                }
                Rule::skip_clause => {
                    skip = Some(self.parse_skip_clause(inner)?);
                }
                Rule::limit_clause => {
                    limit = Some(self.parse_limit_clause(inner)?);
                }
                _ => {}
            }
        }

        let return_clause =
            return_clause.ok_or_else(|| CypherError::syntax(0, 0, "Missing RETURN clause"))?;

        Ok(Query {
            match_clause,
            optional_matches,
            where_clause,
            with_clause: None,
            return_clause,
            order_by,
            skip,
            limit,
        })
    }

    /// Parse MATCH clause
    fn parse_match_clause(&self, pair: Pair<Rule>) -> CypherResult<MatchClause> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern_list {
                patterns = self.parse_pattern_list(inner)?;
            }
        }

        Ok(MatchClause { patterns })
    }

    /// Parse OPTIONAL MATCH
    fn parse_optional_match(&self, pair: Pair<Rule>) -> CypherResult<MatchClause> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern_list {
                patterns = self.parse_pattern_list(inner)?;
            }
        }

        Ok(MatchClause { patterns })
    }

    /// Parse pattern list
    fn parse_pattern_list(&self, pair: Pair<Rule>) -> CypherResult<Vec<Pattern>> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern {
                patterns.push(self.parse_pattern(inner)?);
            }
        }

        Ok(patterns)
    }

    /// Parse a single pattern
    fn parse_pattern(&self, pair: Pair<Rule>) -> CypherResult<Pattern> {
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
    fn parse_node_pattern(&self, pair: Pair<Rule>) -> CypherResult<NodePattern> {
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
    fn parse_relationship_pattern(&self, pair: Pair<Rule>) -> CypherResult<RelPattern> {
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
    fn parse_path_length(&self, pair: Pair<Rule>) -> CypherResult<PathLength> {
        let mut min = None;
        let mut max = None;

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::path_range {
                let range_str = inner.as_str();
                if range_str.contains("..") {
                    let parts: Vec<&str> = range_str.split("..").collect();
                    if !parts[0].is_empty() {
                        min = Some(parts[0].parse().unwrap_or(1));
                    }
                    if parts.len() > 1 && !parts[1].is_empty() {
                        max = Some(parts[1].parse().unwrap_or(u32::MAX));
                    }
                } else {
                    let n: u32 = range_str.parse().unwrap_or(1);
                    min = Some(n);
                    max = Some(n);
                }
            }
        }

        Ok(PathLength { min, max })
    }

    /// Parse WHERE clause
    fn parse_where_clause(&self, pair: Pair<Rule>) -> CypherResult<WhereClause> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::expression {
                return Ok(WhereClause::new(self.parse_expression(inner)?));
            }
        }
        Err(CypherError::syntax(0, 0, "Empty WHERE clause"))
    }

    /// Parse RETURN clause
    fn parse_return_clause(&self, pair: Pair<Rule>) -> CypherResult<ReturnClause> {
        let mut distinct = false;
        let mut items = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::distinct => {
                    distinct = true;
                }
                Rule::return_items => {
                    for item in inner.into_inner() {
                        if item.as_rule() == Rule::return_item {
                            items.push(self.parse_return_item(item)?);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(ReturnClause { distinct, items })
    }

    /// Parse return item
    fn parse_return_item(&self, pair: Pair<Rule>) -> CypherResult<ProjectionItem> {
        let mut expr = None;
        let mut alias = None;

        let content = pair.as_str().trim();
        if content == "*" {
            return Ok(ProjectionItem::all());
        }

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::expression => {
                    expr = Some(self.parse_expression(inner)?);
                }
                Rule::alias => {
                    for alias_inner in inner.into_inner() {
                        if alias_inner.as_rule() == Rule::identifier {
                            alias = Some(String::from(alias_inner.as_str()));
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(ProjectionItem {
            expr: expr.unwrap_or(Expr::Star),
            alias,
        })
    }

    /// Parse ORDER BY clause
    fn parse_order_by_clause(&self, pair: Pair<Rule>) -> CypherResult<OrderByClause> {
        let mut items = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::order_item {
                items.push(self.parse_order_item(inner)?);
            }
        }

        Ok(OrderByClause { items })
    }

    /// Parse order item
    fn parse_order_item(&self, pair: Pair<Rule>) -> CypherResult<OrderByItem> {
        let mut expr = None;
        let mut ascending = true;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::expression => {
                    expr = Some(self.parse_expression(inner)?);
                }
                Rule::order_direction => {
                    let dir = inner.as_str().to_uppercase();
                    ascending = !dir.starts_with("DESC");
                }
                _ => {}
            }
        }

        Ok(OrderByItem {
            expr: expr.ok_or_else(|| CypherError::syntax(0, 0, "Missing ORDER BY expression"))?,
            ascending,
        })
    }

    /// Parse SKIP clause
    fn parse_skip_clause(&self, pair: Pair<Rule>) -> CypherResult<u64> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::integer {
                return inner
                    .as_str()
                    .parse()
                    .map_err(|_| CypherError::syntax(0, 0, "Invalid SKIP value"));
            }
        }
        Err(CypherError::syntax(0, 0, "Missing SKIP value"))
    }

    /// Parse LIMIT clause
    fn parse_limit_clause(&self, pair: Pair<Rule>) -> CypherResult<u64> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::integer {
                return inner
                    .as_str()
                    .parse()
                    .map_err(|_| CypherError::syntax(0, 0, "Invalid LIMIT value"));
            }
        }
        Err(CypherError::syntax(0, 0, "Missing LIMIT value"))
    }

    /// Parse CREATE statement
    fn parse_create_statement(&self, pair: Pair<Rule>) -> CypherResult<CreateClause> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern_list {
                patterns = self.parse_pattern_list(inner)?;
            }
        }

        Ok(CreateClause { patterns })
    }

    /// Parse DELETE statement
    fn parse_delete_statement(&self, pair: Pair<Rule>) -> CypherResult<DeleteClause> {
        let mut detach = false;
        let mut variables = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::detach => {
                    detach = true;
                }
                Rule::variable_list => {
                    for var in inner.into_inner() {
                        if var.as_rule() == Rule::identifier {
                            variables.push(String::from(var.as_str()));
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(DeleteClause { detach, variables })
    }

    /// Parse expression
    fn parse_expression(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // expression = { or_expr }
        // Extract the or_expr child and parse it
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::or_expr {
                return self.parse_or_expr(inner);
            }
        }
        Err(CypherError::syntax(0, 0, "Empty expression"))
    }

    /// Parse OR expression
    fn parse_or_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // or_expr = { xor_expr ~ (^"OR" ~ xor_expr)* }
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty OR expression"));
        }

        // Find all xor_expr children
        let xor_exprs: Vec<_> = inner
            .iter()
            .filter(|p| p.as_rule() == Rule::xor_expr)
            .collect();

        if xor_exprs.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty OR expression"));
        }

        let mut result = self.parse_xor_expr(xor_exprs[0].clone())?;

        for xor_expr in &xor_exprs[1..] {
            let right = self.parse_xor_expr((*xor_expr).clone())?;
            result = Expr::binary(result, BinaryOp::Or, right);
        }

        Ok(result)
    }

    /// Parse XOR expression
    fn parse_xor_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // xor_expr = { and_expr ~ (^"XOR" ~ and_expr)* }
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty XOR expression"));
        }

        // Find all and_expr children
        let and_exprs: Vec<_> = inner
            .iter()
            .filter(|p| p.as_rule() == Rule::and_expr)
            .collect();

        if and_exprs.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty XOR expression"));
        }

        let mut result = self.parse_and_expr(and_exprs[0].clone())?;

        for and_expr in &and_exprs[1..] {
            let right = self.parse_and_expr((*and_expr).clone())?;
            result = Expr::binary(result, BinaryOp::Xor, right);
        }

        Ok(result)
    }

    /// Parse AND expression
    fn parse_and_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // and_expr = { not_expr ~ (^"AND" ~ not_expr)* }
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty AND expression"));
        }

        // Find all not_expr children
        let not_exprs: Vec<_> = inner
            .iter()
            .filter(|p| p.as_rule() == Rule::not_expr)
            .collect();

        if not_exprs.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty AND expression"));
        }

        let mut result = self.parse_not_expr(not_exprs[0].clone())?;

        for not_expr in &not_exprs[1..] {
            let right = self.parse_not_expr((*not_expr).clone())?;
            result = Expr::binary(result, BinaryOp::And, right);
        }

        Ok(result)
    }

    /// Parse NOT expression
    fn parse_not_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // not_expr = { ^"NOT"* ~ comparison }
        let mut not_count = 0;
        let mut comparison = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::comparison => {
                    comparison = Some(self.parse_comparison(inner)?);
                }
                _ => {
                    // Count NOT keywords
                    if inner.as_str().to_uppercase() == "NOT" {
                        not_count += 1;
                    }
                }
            }
        }

        let mut result =
            comparison.ok_or_else(|| CypherError::syntax(0, 0, "Missing comparison"))?;

        // Apply NOT operators (odd number = NOT, even = no-op)
        if not_count % 2 == 1 {
            result = Expr::unary(UnaryOp::Not, result);
        }

        Ok(result)
    }

    /// Parse comparison expression
    fn parse_comparison(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty comparison"));
        }

        let mut result = self.parse_addition(inner[0].clone())?;

        let mut i = 1;
        while i < inner.len() {
            if inner[i].as_rule() == Rule::comparison_op {
                let op_str = inner[i].as_str().to_uppercase();
                let op = self.parse_comparison_op(&op_str)?;

                if i + 1 < inner.len() {
                    let right = self.parse_addition(inner[i + 1].clone())?;

                    // Handle special operators
                    match op_str.as_str() {
                        s if s.contains("NULL") => {
                            result = Expr::IsNull {
                                expr: Box::new(result),
                                negated: s.contains("NOT"),
                            };
                        }
                        s if s.contains("IN") => {
                            result = Expr::In {
                                expr: Box::new(result),
                                list: Box::new(right),
                                negated: false,
                            };
                        }
                        s if s.contains("STARTS") => {
                            result = Expr::StringPredicate {
                                expr: Box::new(result),
                                predicate: StringPredicateType::StartsWith,
                                pattern: Box::new(right),
                            };
                        }
                        s if s.contains("ENDS") => {
                            result = Expr::StringPredicate {
                                expr: Box::new(result),
                                predicate: StringPredicateType::EndsWith,
                                pattern: Box::new(right),
                            };
                        }
                        s if s.contains("CONTAINS") => {
                            result = Expr::StringPredicate {
                                expr: Box::new(result),
                                predicate: StringPredicateType::Contains,
                                pattern: Box::new(right),
                            };
                        }
                        _ => {
                            result = Expr::binary(result, op, right);
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        Ok(result)
    }

    /// Parse comparison operator
    fn parse_comparison_op(&self, op: &str) -> CypherResult<BinaryOp> {
        match op {
            "=" => Ok(BinaryOp::Eq),
            "<>" | "!=" => Ok(BinaryOp::Neq),
            "<" => Ok(BinaryOp::Lt),
            "<=" => Ok(BinaryOp::Lte),
            ">" => Ok(BinaryOp::Gt),
            ">=" => Ok(BinaryOp::Gte),
            _ => Ok(BinaryOp::Eq), // Fallback for special operators handled separately
        }
    }

    /// Parse addition expression
    fn parse_addition(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty addition"));
        }

        let mut result = self.parse_multiplication(inner[0].clone())?;

        let mut i = 1;
        while i < inner.len() {
            let token = inner[i].as_str();
            if token == "+" || token == "-" {
                if i + 1 < inner.len() {
                    let right = self.parse_multiplication(inner[i + 1].clone())?;
                    let op = if token == "+" {
                        BinaryOp::Add
                    } else {
                        BinaryOp::Sub
                    };
                    result = Expr::binary(result, op, right);
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                // Try to parse as multiplication
                let right = self.parse_multiplication(inner[i].clone())?;
                result = Expr::binary(result, BinaryOp::Add, right);
                i += 1;
            }
        }

        Ok(result)
    }

    /// Parse multiplication expression
    fn parse_multiplication(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty multiplication"));
        }

        let mut result = self.parse_power(inner[0].clone())?;

        let mut i = 1;
        while i < inner.len() {
            let token = inner[i].as_str();
            if token == "*" || token == "/" || token == "%" {
                if i + 1 < inner.len() {
                    let right = self.parse_power(inner[i + 1].clone())?;
                    let op = match token {
                        "*" => BinaryOp::Mul,
                        "/" => BinaryOp::Div,
                        "%" => BinaryOp::Mod,
                        // Guard against impossible parse state - should never occur
                        other => {
                            return Err(CypherError::internal(format!(
                                "Unexpected operator in multiplication: {}",
                                other
                            )))
                        }
                    };
                    result = Expr::binary(result, op, right);
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                let right = self.parse_power(inner[i].clone())?;
                result = Expr::binary(result, BinaryOp::Mul, right);
                i += 1;
            }
        }

        Ok(result)
    }

    /// Parse power expression
    fn parse_power(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty power"));
        }

        let mut result = self.parse_unary(inner[0].clone())?;

        for item in inner[1..].iter() {
            if item.as_str() == "^" {
                continue;
            }
            let right = self.parse_unary(item.clone())?;
            result = Expr::binary(result, BinaryOp::Pow, right);
        }

        Ok(result)
    }

    /// Parse unary expression
    fn parse_unary(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut neg = false;
        let mut pos = false;
        let mut expr_pair = None;

        for inner in pair.into_inner() {
            match inner.as_str() {
                "-" => neg = true,
                "+" => pos = true,
                _ => {
                    if inner.as_rule() == Rule::property_or_atom {
                        expr_pair = Some(inner);
                    }
                }
            }
        }

        let expr = if let Some(p) = expr_pair {
            self.parse_property_or_atom(p)?
        } else {
            return Err(CypherError::syntax(0, 0, "Missing unary operand"));
        };

        if neg {
            Ok(Expr::unary(UnaryOp::Neg, expr))
        } else if pos {
            Ok(Expr::unary(UnaryOp::Pos, expr))
        } else {
            Ok(expr)
        }
    }

    /// Parse property access or atom
    fn parse_property_or_atom(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut result = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::atom => {
                    result = Some(self.parse_atom(inner)?);
                }
                Rule::property_access => {
                    for prop_inner in inner.into_inner() {
                        if prop_inner.as_rule() == Rule::identifier {
                            result = Some(Expr::property(
                                result.take().ok_or_else(|| {
                                    CypherError::syntax(0, 0, "Property access without base")
                                })?,
                                prop_inner.as_str(),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        result.ok_or_else(|| CypherError::syntax(0, 0, "Empty property or atom"))
    }

    /// Parse atom
    fn parse_atom(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::literal => {
                    return Ok(Expr::Literal(self.parse_literal(inner)?));
                }
                Rule::parameter => {
                    return self.parse_parameter(inner);
                }
                Rule::function_call => {
                    return self.parse_function_call(inner);
                }
                Rule::case_expr => {
                    return self.parse_case_expr(inner);
                }
                Rule::list_literal => {
                    return self.parse_list_literal(inner);
                }
                Rule::map_literal => {
                    return Ok(Expr::Map(self.parse_map_literal(inner)?));
                }
                Rule::expression => {
                    return self.parse_expression(inner);
                }
                Rule::identifier => {
                    return Ok(Expr::variable(inner.as_str()));
                }
                _ => {}
            }
        }

        Err(CypherError::syntax(0, 0, "Invalid atom"))
    }

    /// Parse literal value
    fn parse_literal(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::null_literal => {
                    return Ok(CypherValue::Null);
                }
                Rule::boolean_literal => {
                    let val = inner.as_str().to_uppercase() == "TRUE";
                    return Ok(CypherValue::Bool(val));
                }
                Rule::number_literal => {
                    return self.parse_number(inner);
                }
                Rule::string_literal => {
                    return self.parse_string_literal(inner);
                }
                _ => {}
            }
        }

        Err(CypherError::syntax(0, 0, "Invalid literal"))
    }

    /// Parse number literal
    fn parse_number(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        // Save string before consuming pair
        let s = pair.as_str().to_string();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::float => {
                    let val: f64 = inner
                        .as_str()
                        .parse()
                        .map_err(|_| CypherError::syntax(0, 0, "Invalid float literal"))?;
                    return Ok(CypherValue::Float(val));
                }
                Rule::integer => {
                    let val: i64 = inner
                        .as_str()
                        .parse()
                        .map_err(|_| CypherError::syntax(0, 0, "Invalid integer literal"))?;
                    return Ok(CypherValue::Int(val));
                }
                _ => {}
            }
        }

        // Try parsing the string directly
        let s = s.as_str();
        if s.contains('.') {
            let val: f64 = s
                .parse()
                .map_err(|_| CypherError::syntax(0, 0, "Invalid float literal"))?;
            Ok(CypherValue::Float(val))
        } else {
            let val: i64 = s
                .parse()
                .map_err(|_| CypherError::syntax(0, 0, "Invalid integer literal"))?;
            Ok(CypherValue::Int(val))
        }
    }

    /// Parse string literal
    fn parse_string_literal(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        let s = pair.as_str();
        // Remove quotes and handle escape sequences
        let inner = if s.starts_with('"') {
            &s[1..s.len() - 1]
        } else if s.starts_with('\'') {
            &s[1..s.len() - 1]
        } else {
            s
        };

        // Unescape
        let unescaped = inner
            .replace("\\\"", "\"")
            .replace("\\'", "'")
            .replace("\\\\", "\\")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t");

        Ok(CypherValue::string(unescaped))
    }

    /// Parse parameter
    fn parse_parameter(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // Save string before consuming pair
        let s = pair.as_str().to_string();
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::identifier {
                return Ok(Expr::Parameter(String::from(inner.as_str())));
            }
        }
        // Fallback: extract parameter name from the pair itself
        if s.starts_with('$') {
            Ok(Expr::Parameter(String::from(&s[1..])))
        } else {
            Err(CypherError::syntax(0, 0, "Invalid parameter"))
        }
    }

    /// Parse function call
    fn parse_function_call(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut name = None;
        let mut args = Vec::new();
        let mut distinct = false;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier => {
                    if name.is_none() {
                        name = Some(String::from(inner.as_str()));
                    }
                }
                Rule::distinct => {
                    distinct = true;
                }
                Rule::expression => {
                    args.push(self.parse_expression(inner)?);
                }
                _ => {}
            }
        }

        Ok(Expr::FunctionCall {
            name: name.ok_or_else(|| CypherError::syntax(0, 0, "Missing function name"))?,
            args,
            distinct,
        })
    }

    /// Parse case expression
    fn parse_case_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut operand = None;
        let mut when_clauses = Vec::new();
        let mut else_clause = None;

        let current_when: Option<Expr> = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::expression if operand.is_none() && current_when.is_none() => {
                    operand = Some(Box::new(self.parse_expression(inner)?));
                }
                Rule::when_clause => {
                    let mut when_expr = None;
                    let mut then_expr = None;
                    for when_inner in inner.into_inner() {
                        if when_inner.as_rule() == Rule::expression {
                            if when_expr.is_none() {
                                when_expr = Some(self.parse_expression(when_inner)?);
                            } else {
                                then_expr = Some(self.parse_expression(when_inner)?);
                            }
                        }
                    }
                    if let (Some(w), Some(t)) = (when_expr, then_expr) {
                        when_clauses.push((w, t));
                    }
                }
                Rule::else_clause => {
                    for else_inner in inner.into_inner() {
                        if else_inner.as_rule() == Rule::expression {
                            else_clause = Some(Box::new(self.parse_expression(else_inner)?));
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Expr::Case {
            operand,
            when_clauses,
            else_clause,
        })
    }

    /// Parse list literal
    fn parse_list_literal(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut items = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::expression {
                items.push(self.parse_expression(inner)?);
            }
        }

        Ok(Expr::List(items))
    }

    /// Parse map literal (also used for properties)
    fn parse_map_literal(&self, pair: Pair<Rule>) -> CypherResult<MapLiteral> {
        let mut entries = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::property_list | Rule::map_entry | Rule::property => {
                    // Handle both property_list and direct map_entry
                    for entry in inner.into_inner() {
                        match entry.as_rule() {
                            Rule::property | Rule::map_entry => {
                                let mut key = None;
                                let mut value = None;
                                for e in entry.into_inner() {
                                    match e.as_rule() {
                                        Rule::identifier => {
                                            key = Some(String::from(e.as_str()));
                                        }
                                        Rule::expression => {
                                            value = Some(self.parse_expression(e)?);
                                        }
                                        _ => {}
                                    }
                                }
                                if let (Some(k), Some(v)) = (key, value) {
                                    entries.push((k, Box::new(v)));
                                }
                            }
                            Rule::identifier => {
                                // First identifier in a property
                                let _key = String::from(entry.as_str());
                                // Look for the next sibling (value)
                                // This branch handles direct identifier tokens
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(MapLiteral { entries })
    }

    /// Convert pest error to CypherError
    fn pest_error_to_cypher(&self, error: pest::error::Error<Rule>) -> CypherError {
        let (line, column) = match error.line_col {
            pest::error::LineColLocation::Pos((l, c)) => (l, c),
            pest::error::LineColLocation::Span((l, c), _) => (l, c),
        };

        CypherError::SyntaxError {
            line,
            column,
            message: error.to_string(),
        }
    }
}

impl Default for CypherParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_parse_simple_match() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n) RETURN n");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Query(query)) = result {
            assert!(query.match_clause.is_some());
            assert_eq!(query.return_clause.items.len(), 1);
        }
    }

    #[test]
    fn test_parse_match_with_label() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n:Person) RETURN n");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Query(query)) = result {
            let match_clause = query.match_clause.unwrap();
            if let PatternElement::Node(node) = &match_clause.patterns[0].elements[0] {
                assert_eq!(node.labels[0], "Person");
            }
        }
    }

    #[test]
    fn test_parse_match_with_properties() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n:Person {name: \"Alice\"}) RETURN n");
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_parse_match_with_relationship() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (a)-[r:KNOWS]->(b) RETURN a, b");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Query(query)) = result {
            let match_clause = query.match_clause.unwrap();
            assert_eq!(match_clause.patterns[0].elements.len(), 3);
        }
    }

    #[test]
    fn test_parse_where_clause() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n:Person) WHERE n.age > 25 RETURN n");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Query(query)) = result {
            assert!(query.where_clause.is_some());
        }
    }

    #[test]
    fn test_parse_order_by() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n) RETURN n ORDER BY n.name");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Query(query)) = result {
            assert!(query.order_by.is_some());
        }
    }

    #[test]
    fn test_parse_limit() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n) RETURN n LIMIT 10");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Query(query)) = result {
            assert_eq!(query.limit, Some(10));
        }
    }

    #[test]
    fn test_parse_create_node() {
        let parser = CypherParser::new();
        let result = parser.parse("CREATE (n:Person {name: \"Alice\"})");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Create(create)) = result {
            assert_eq!(create.patterns.len(), 1);
        }
    }

    #[test]
    fn test_parse_delete() {
        let parser = CypherParser::new();
        let result = parser.parse("DELETE n");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Delete(delete)) = result {
            assert!(!delete.detach);
            assert_eq!(delete.variables.len(), 1);
        }
    }

    #[test]
    fn test_parse_detach_delete() {
        let parser = CypherParser::new();
        let result = parser.parse("DETACH DELETE n");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Delete(delete)) = result {
            assert!(delete.detach);
        }
    }

    #[test]
    fn test_parse_return_alias() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n) RETURN n.name AS name");
        assert!(result.is_ok(), "Failed: {:?}", result);

        if let Ok(Statement::Query(query)) = result {
            assert!(query.return_clause.items[0].alias.is_some());
        }
    }

    #[test]
    fn test_parse_complex_where() {
        let parser = CypherParser::new();
        let result =
            parser.parse("MATCH (n:Person) WHERE n.age > 25 AND n.name = \"Alice\" RETURN n");
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_parse_function_call() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH (n) RETURN count(n)");
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_parse_invalid_syntax() {
        let parser = CypherParser::new();
        let result = parser.parse("MATCH RETURN");
        assert!(result.is_err());
    }
}
