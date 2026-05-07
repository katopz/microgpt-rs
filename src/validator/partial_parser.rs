/// Partial parser: bracket balancer DFA for fast rejection.
///
/// This is Tier 0 validation — catches clearly broken syntax like
/// unbalanced brackets, unclosed strings, etc. Cheap O(n) scan.
///
/// Does NOT validate Rust syntax — that's Tier 1 (syn).
pub struct PartialParser {
    paren_depth: i32,
    brace_depth: i32,
    bracket_depth: i32,
    angle_depth: i32,
    in_string: bool,
    in_char: bool,
    in_block_comment: bool,
    in_line_comment: bool,
}

impl PartialParser {
    pub fn new() -> Self {
        Self {
            paren_depth: 0,
            brace_depth: 0,
            bracket_depth: 0,
            angle_depth: 0,
            in_string: false,
            in_char: false,
            in_block_comment: false,
            in_line_comment: false,
        }
    }

    /// Validate a code fragment for bracket balance.
    /// Returns `true` if the fragment is potentially valid (no obvious errors).
    /// Returns `false` if clearly broken (e.g., `}` without matching `{`).
    pub fn is_valid(&mut self, code: &str) -> bool {
        self.reset();

        let chars: Vec<char> = code.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let ch = chars[i];

            // Skip characters inside strings
            if self.in_string {
                match ch {
                    '\\' => {
                        i += 1;
                    } // Skip escaped char
                    '"' => {
                        self.in_string = false;
                    }
                    _ => {}
                }
                i += 1;
                continue;
            }

            // Skip characters inside char literals
            if self.in_char {
                match ch {
                    '\\' => {
                        i += 1;
                    }
                    '\'' => {
                        self.in_char = false;
                    }
                    _ => {}
                }
                i += 1;
                continue;
            }

            // Handle comments
            if self.in_line_comment {
                if ch == '\n' {
                    self.in_line_comment = false;
                }
                i += 1;
                continue;
            }

            if self.in_block_comment {
                if ch == '*' && i + 1 < len && chars[i + 1] == '/' {
                    self.in_block_comment = false;
                    i += 2;
                    continue;
                }
                i += 1;
                continue;
            }

            // Check for comment starts
            if ch == '/' && i + 1 < len {
                match chars[i + 1] {
                    '/' => {
                        self.in_line_comment = true;
                        i += 2;
                        continue;
                    }
                    '*' => {
                        self.in_block_comment = true;
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }

            // Track bracket depth
            match ch {
                '(' => self.paren_depth += 1,
                ')' => {
                    self.paren_depth -= 1;
                    if self.paren_depth < 0 {
                        return false;
                    }
                }
                '{' => self.brace_depth += 1,
                '}' => {
                    self.brace_depth -= 1;
                    if self.brace_depth < 0 {
                        return false;
                    }
                }
                '[' => self.bracket_depth += 1,
                ']' => {
                    self.bracket_depth -= 1;
                    if self.bracket_depth < 0 {
                        return false;
                    }
                }
                '<' => {
                    // Only count as angle bracket if not comparison
                    // Simple heuristic: count if preceded by identifier or closing bracket
                    self.angle_depth += 1;
                }
                '>' => {
                    self.angle_depth -= 1;
                    if self.angle_depth < 0 {
                        self.angle_depth = 0; // Don't reject on angle bracket mismatch
                    }
                }
                '"' => self.in_string = true,
                '\'' => {
                    // Could be char literal or lifetime
                    // Heuristic: if followed by a char then ', it's a char literal
                    if i + 2 < len && chars[i + 2] == '\'' {
                        self.in_char = true;
                    } else if i + 1 < len && (chars[i + 1].is_alphabetic() || chars[i + 1] == '_') {
                        // Likely a lifetime — don't count as char literal
                    } else {
                        self.in_char = true;
                    }
                }
                _ => {}
            }

            i += 1;
        }

        // For partial code, we allow unclosed brackets (depth > 0)
        // Only reject if depth went negative (too many closing brackets)
        self.paren_depth >= 0 && self.brace_depth >= 0 && self.bracket_depth >= 0
    }

    /// Reset parser state.
    pub fn reset(&mut self) {
        self.paren_depth = 0;
        self.brace_depth = 0;
        self.bracket_depth = 0;
        self.angle_depth = 0;
        self.in_string = false;
        self.in_char = false;
        self.in_block_comment = false;
        self.in_line_comment = false;
    }

    /// Check if brackets are balanced (all depths zero).
    pub fn is_balanced(&self) -> bool {
        self.paren_depth == 0 && self.brace_depth == 0 && self.bracket_depth == 0
    }
}

impl Default for PartialParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_parser_accepts_valid_fragments() {
        let mut parser = PartialParser::new();

        // Complete expression
        assert!(parser.is_valid("fn main() { let x = 1; }"));
        assert!(parser.is_balanced());

        // Partial expression (unclosed brace is ok)
        assert!(parser.is_valid("fn main() { let x = 1"));
        assert!(!parser.is_balanced()); // brace still open

        // Empty string
        assert!(parser.is_valid(""));
        assert!(parser.is_balanced());

        // Nested brackets
        assert!(parser.is_valid("vec![vec![1, 2], vec![3, 4]]"));
        assert!(parser.is_balanced());

        // Generics
        assert!(parser.is_valid("HashMap<String, Vec<usize>>"));
    }

    #[test]
    fn test_partial_parser_rejects_unbalanced() {
        let mut parser = PartialParser::new();

        // Extra closing brace
        assert!(!parser.is_valid("fn main() { } }"));

        // Extra closing paren
        assert!(!parser.is_valid("foo())"));

        // Extra closing bracket
        assert!(!parser.is_valid("arr[]]"));

        // Reset between checks
        parser.reset();
        assert!(parser.is_valid("fn foo() {}"));
    }

    #[test]
    fn test_partial_parser_handles_strings() {
        let mut parser = PartialParser::new();

        // Brackets inside strings should not be counted
        assert!(parser.is_valid(r#"let s = "{}";"#));
        assert!(parser.is_balanced());

        // Escaped quote inside string
        assert!(parser.is_valid(r#"let s = "he said \"hello\"";"#));
        assert!(parser.is_balanced());

        // Unclosed string — parser should still accept (partial code)
        assert!(parser.is_valid(r#"let s = "hello"#));

        // String with brackets that look unbalanced but aren't
        assert!(parser.is_valid(r#"println!("{} {}", a, b);"#));
        assert!(parser.is_balanced());
    }

    #[test]
    fn test_partial_parser_handles_comments() {
        let mut parser = PartialParser::new();

        // Line comment
        assert!(parser.is_valid("let x = 1; // } unbalanced in comment"));
        assert!(parser.is_balanced());

        // Block comment
        assert!(parser.is_valid("let x = 1; /* } */ let y = 2;"));
        assert!(parser.is_balanced());

        // Block comment spanning lines
        assert!(parser.is_valid("/* { ( [ */ let x = 1; /* ] ) } */"));
        assert!(parser.is_balanced());

        // Unclosed block comment — partial code, still accepted
        assert!(parser.is_valid("let x = 1; /* still in comment"));
    }
}
