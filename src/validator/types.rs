/// Result of pruning a token sequence.
#[derive(Debug, Clone)]
pub struct PruneResult {
    pub is_valid: bool,
    pub error_kind: ErrorKind,
}

/// Category of syntax error found during pruning.
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    /// No error — the sequence is valid.
    None,
    /// Unbalanced brackets: mismatched `{`, `(`, `[`, `<`.
    UnbalancedBrackets,
    /// `syn` parse error — invalid Rust syntax.
    SynError(String),
}

/// Feedback from the compiler for self-correction.
#[derive(Debug, Clone)]
pub struct CompilerFeedback {
    /// The error message from the compiler.
    pub error_message: String,
    /// The code fragment that caused the error.
    pub failing_code: String,
    /// Suggested fix (if any).
    pub suggestion: Option<String>,
}

impl CompilerFeedback {
    /// Extract a suggestion from a syn error message.
    pub fn extract_suggestion(error_msg: &str) -> Option<String> {
        // syn errors sometimes contain "expected X" patterns
        if error_msg.contains("expected") {
            let start = error_msg.find("expected")?;
            Some(error_msg[start..].to_string())
        } else {
            None
        }
    }

    /// Convert to context string for inclusion in prompt.
    pub fn to_context(&self) -> String {
        let mut ctx = format!("Error: {}", self.error_message);
        if let Some(suggestion) = &self.suggestion {
            ctx.push_str(&format!("\nSuggestion: {suggestion}"));
        }
        ctx
    }
}
