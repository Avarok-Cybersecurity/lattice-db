//! Naming constructs outside the grammar.
//!
//! Only consulted after the grammar rejected a query, so it can never change how a valid query
//! parses. It scans tokens outside string literals, backtick identifiers and comments, and names
//! the first construct the engine does not implement (`MERGE`, `SET`, list comprehensions, ...)
//! instead of reporting a bare syntax error at a column.

use crate::cypher::error::CypherError;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),
    Punct(char),
}

/// Tokens of `input`, with literals, quoted identifiers and comments removed
fn tokens(input: &str) -> Vec<Token> {
    let chars: Vec<char> = input.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\'' | '"' | '`' => {
                i += 1;
                while i < chars.len() && chars[i] != c {
                    if chars[i] == '\\' && c != '`' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
                // A quoted identifier is still an identifier-shaped token.
                if c == '`' {
                    out.push(Token::Word(String::new()));
                }
            }
            '/' if chars.get(i + 1) == Some(&'/') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if chars.get(i + 1) == Some(&'*') => {
                i += 2;
                while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                    i += 1;
                }
                i += 2;
            }
            c if c.is_alphanumeric() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                out.push(Token::Word(chars[start..i].iter().collect()));
            }
            c if c.is_whitespace() => i += 1,
            c => {
                out.push(Token::Punct(c));
                i += 1;
            }
        }
    }
    out
}

/// Maximum depth of `(`, `[` and `{` nesting outside literals and comments
pub(super) fn nesting_depth(input: &str) -> usize {
    let (mut depth, mut max) = (0usize, 0usize);
    for t in tokens(input) {
        match t {
            Token::Punct('(' | '[' | '{') => {
                depth += 1;
                max = max.max(depth);
            }
            Token::Punct(')' | ']' | '}') => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    max
}

const CLAUSES: [&str; 14] = [
    "MERGE", "SET", "REMOVE", "CALL", "FOREACH", "LOAD", "UNION", "DROP", "CREATE", "DELETE",
    "DETACH", "USE", "SHOW", "FINISH",
];

fn word(t: Option<&Token>) -> Option<&str> {
    match t {
        Some(Token::Word(w)) => Some(w.as_str()),
        _ => None,
    }
}

fn punct(t: Option<&Token>, c: char) -> bool {
    t == Some(&Token::Punct(c))
}

fn is_kw(t: Option<&Token>, kw: &str) -> bool {
    word(t).is_some_and(|w| w.eq_ignore_ascii_case(kw))
}

/// Name the first unsupported construct in a query the grammar rejected
pub(super) fn diagnose(input: &str) -> Option<CypherError> {
    let t = tokens(input);
    for (i, tok) in t.iter().enumerate() {
        let prev = i.checked_sub(1).and_then(|p| t.get(p));
        let next = t.get(i + 1);
        // `n.set`, `:SET` and `$set` are a property, a label and a parameter, not clauses.
        let qualified = punct(prev, '.') || punct(prev, ':') || punct(prev, '$');
        match tok {
            Token::Word(w) if !qualified => {
                let upper = w.to_ascii_uppercase();
                if (upper == "SHORTESTPATH" || upper == "ALLSHORTESTPATHS") && punct(next, '(') {
                    return Some(CypherError::unsupported(format!("{w}() path search")));
                }
                if (upper == "EXISTS" || upper == "COUNT" || upper == "COLLECT") && punct(next, '{')
                {
                    return Some(CypherError::unsupported(format!(
                        "{upper} {{ … }} subquery"
                    )));
                }
                if upper == "REDUCE" && punct(next, '(') {
                    return Some(CypherError::unsupported("reduce()"));
                }
                if matches!(upper.as_str(), "ANY" | "ALL" | "NONE" | "SINGLE")
                    && punct(next, '(')
                    && is_kw(t.get(i + 3), "IN")
                {
                    return Some(CypherError::unsupported(format!(
                        "list predicate function {w}()"
                    )));
                }
                if CLAUSES.contains(&upper.as_str()) {
                    return Some(CypherError::UnsupportedClause { clause: upper });
                }
                // MATCH p = (…) / , p = (…)
                if punct(next, '=')
                    && punct(t.get(i + 2), '(')
                    && (is_kw(prev, "MATCH") || punct(prev, ','))
                {
                    return Some(CypherError::unsupported(format!("named path ({w} = …)")));
                }
            }
            Token::Punct('[') if word(next).is_some() && is_kw(t.get(i + 2), "IN") => {
                return Some(CypherError::unsupported("list comprehension"));
            }
            Token::Punct('[') if punct(next, '(') => {
                // `[(a)-->(b) | …]`: a pattern right after the opening bracket
                let close = t[i + 1..].iter().position(|x| *x == Token::Punct(')'));
                if let Some(c) = close {
                    if matches!(t.get(i + 2 + c), Some(Token::Punct('-' | '<'))) {
                        return Some(CypherError::unsupported("pattern comprehension"));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn named(q: &str) -> String {
        diagnose(q).map(|e| e.to_string()).unwrap_or_default()
    }

    #[test]
    fn words_inside_literals_comments_and_qualified_names_are_not_clauses() {
        assert_eq!(named("RETURN 'MERGE' + \"SET\" // CALL\n"), "");
        assert_eq!(named("RETURN n.set, $merge /* UNION */"), "");
        assert_eq!(named("MATCH (n:SET) RETURN `CALL`"), "");
    }

    #[test]
    fn the_first_unsupported_construct_is_named() {
        assert_eq!(
            named("MATCH (n) SET n.x = 1 MERGE (m)"),
            "Unsupported clause: SET"
        );
        assert_eq!(
            named("RETURN [x IN [1] | x]"),
            "Unsupported feature: list comprehension"
        );
        assert_eq!(
            named("RETURN [(a)-->(b) | b]"),
            "Unsupported feature: pattern comprehension"
        );
        assert_eq!(
            named("MATCH p = (a)-->(b) RETURN p"),
            "Unsupported feature: named path (p = …)"
        );
        assert_eq!(
            named("RETURN any(x IN [1] WHERE x > 0)"),
            "Unsupported feature: list predicate function any()"
        );
    }

    #[test]
    fn nesting_depth_ignores_brackets_in_literals() {
        assert_eq!(nesting_depth("RETURN [[1], {a: (2)}]"), 3);
        assert_eq!(nesting_depth("RETURN '((((('"), 0);
    }
}
