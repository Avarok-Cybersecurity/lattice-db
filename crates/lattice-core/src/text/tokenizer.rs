//! Deterministic, Unicode-aware tokenizer for the lexical index.
//!
//! Rules, in full:
//! 1. A token is a maximal run of characters for which [`char::is_alphanumeric`]
//!    holds (Unicode `Alphabetic` or `Numeric`). Everything else separates.
//! 2. Each token is lowercased with [`char::to_lowercase`] (full Unicode case
//!    mapping, locale-independent).
//! 3. Tokens longer than [`MAX_TOKEN_BYTES`] are dropped.
//!
//! No stemming, no stop words: both are language-specific, and BM25's IDF
//! already discounts very common terms. Consequences worth knowing: "don't"
//! yields `don` + `t`; a run of CJK ideographs without spaces is one token.

/// Tokens longer than this many UTF-8 bytes are discarded rather than indexed.
///
/// Real words are far shorter; a longer "word" is a hash, a base64 blob or an
/// attack on vocabulary size, and indexing it only grows memory.
pub const MAX_TOKEN_BYTES: usize = 64;

/// Split `text` into lowercase alphanumeric tokens (see the module rules).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            push_token(&mut tokens, std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        push_token(&mut tokens, current);
    }
    tokens
}

fn push_token(tokens: &mut Vec<String>, token: String) {
    if token.len() <= MAX_TOKEN_BYTES {
        tokens.push(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn splits_on_non_alphanumerics_and_lowercases() {
        assert_eq!(
            tokenize("Hello, World! GeoHarness-v2 (beta)"),
            vec!["hello", "world", "geoharness", "v2", "beta"]
        );
    }

    #[test]
    fn unicode_letters_and_case_mapping() {
        assert_eq!(
            tokenize("Ärger ÜBER Straße"),
            vec!["ärger", "über", "straße"]
        );
        assert_eq!(tokenize("ΣΟΦΙΑ σοφία"), vec!["σοφια", "σοφία"]);
        assert_eq!(tokenize("naïve café"), vec!["naïve", "café"]);
    }

    #[test]
    fn empty_and_punctuation_only_inputs_yield_nothing() {
        assert!(tokenize("").is_empty());
        assert!(tokenize(" ... --- !!! ").is_empty());
    }

    #[test]
    fn overlong_tokens_are_dropped() {
        let long = "a".repeat(MAX_TOKEN_BYTES + 1);
        let exact = "b".repeat(MAX_TOKEN_BYTES);
        assert_eq!(
            tokenize(&format!("{long} ok {exact}")),
            vec!["ok".to_string(), exact]
        );
    }

    #[test]
    fn is_deterministic() {
        let text = "The quick brown fox — 123 jumps; über alles.";
        assert_eq!(tokenize(text), tokenize(text));
    }
}
