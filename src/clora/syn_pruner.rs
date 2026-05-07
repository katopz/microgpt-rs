use super::partial_parser::PartialParser;
use super::types::PruneResult;
use crate::speculative::types::ConstraintPruner;
use crate::tokenizer::BpeTokenizer;
use crate::tokenizer::BpeTokenizerImpl;
use std::sync::Arc;

/// Two-tier syntax pruner for cLoRA.
///
/// Tier 0: Bracket balancer DFA (PartialParser) — O(n), rejects clearly broken code.
/// Tier 1: `syn` parse attempt — accurate, but expensive. Only called if Tier 0 passes.
pub struct SynPruner {
    tokenizer: Arc<BpeTokenizer>,
    parser: PartialParser,
}

impl SynPruner {
    pub fn new(tokenizer: Arc<BpeTokenizer>) -> Self {
        Self {
            tokenizer,
            parser: PartialParser::new(),
        }
    }

    /// Validate a complete code string through both tiers.
    pub fn validate(&mut self, code: &str) -> PruneResult {
        // Tier 0: Bracket balance
        if !self.parser.is_valid(code) {
            return PruneResult {
                is_valid: false,
                error_kind: super::types::ErrorKind::UnbalancedBrackets,
            };
        }

        // Tier 1: syn parse
        match syn::parse_str::<syn::Stmt>(code) {
            Ok(_) => PruneResult {
                is_valid: true,
                error_kind: super::types::ErrorKind::None,
            },
            Err(e) => PruneResult {
                is_valid: false,
                error_kind: super::types::ErrorKind::SynError(e.to_string()),
            },
        }
    }

    /// Quick Tier 0 check only (for DDTree hot path).
    pub fn is_valid_quick(&mut self, code: &str) -> bool {
        self.parser.is_valid(code)
    }
}

impl ConstraintPruner for SynPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Decode tokens to string for validation
        let mut all_tokens = parent_tokens.to_vec();
        all_tokens.push(token_idx);

        let code = BpeTokenizerImpl::decode(&self.tokenizer, &all_tokens);

        // Only do Tier 0 (bracket balance) in the hot path.
        // Tier 1 (syn) is too expensive for every DDTree node.
        let mut parser = PartialParser::new();
        parser.is_valid(&code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syn_pruner_accepts_valid_rust() {
        let tokenizer = Arc::new(crate::tokenizer::BpeTrainer::train("fn let mut x", 64));
        let mut pruner = SynPruner::new(tokenizer);

        let result = pruner.validate("let x = 42;");
        assert!(result.is_valid, "expected valid for 'let x = 42;'");
        assert_eq!(result.error_kind, super::super::types::ErrorKind::None);

        let result = pruner.validate("fn main() { }");
        assert!(result.is_valid, "expected valid for 'fn main() {{ }}'");

        let result = pruner.validate("let s = \"hello\";");
        assert!(result.is_valid, "expected valid for string literal");
    }

    #[test]
    fn test_syn_pruner_rejects_invalid_rust() {
        let tokenizer = Arc::new(crate::tokenizer::BpeTrainer::train("fn let mut x", 64));
        let mut pruner = SynPruner::new(tokenizer);

        let result = pruner.validate("let = ;");
        assert!(!result.is_valid, "expected invalid for 'let = ;'");

        match result.error_kind {
            super::super::types::ErrorKind::SynError(msg) => {
                assert!(!msg.is_empty(), "syn error should have a message");
            }
            other => panic!("expected SynError, got {other:?}"),
        }
    }

    #[test]
    fn test_syn_pruner_bracket_tier_rejects() {
        let tokenizer = Arc::new(crate::tokenizer::BpeTrainer::train("fn let { }", 64));
        let mut pruner = SynPruner::new(tokenizer);

        // Unmatched closing brace — Tier 0 should reject before syn sees it
        let result = pruner.validate("fn main() { } }");
        assert!(!result.is_valid, "expected invalid for unbalanced braces");
        assert_eq!(
            result.error_kind,
            super::super::types::ErrorKind::UnbalancedBrackets
        );

        // Unmatched closing paren
        let result = pruner.validate("foo())");
        assert!(!result.is_valid, "expected invalid for unbalanced parens");
        assert_eq!(
            result.error_kind,
            super::super::types::ErrorKind::UnbalancedBrackets
        );
    }
}
