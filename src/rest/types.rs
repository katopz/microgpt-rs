//! REST types for anyrag API communication.
//!
//! Matches anyrag's `/search/vector` endpoint format.
//! Token sequences are stored in document descriptions as comma-separated IDs.

use serde::{Deserialize, Serialize};

/// Request body for anyrag /search/vector endpoint.
#[derive(Debug, Serialize)]
pub struct SearchRequest {
    /// Text query for vector search.
    pub query: String,
    /// Maximum number of results to return.
    pub limit: Option<u32>,
    /// Optional database name.
    pub db: Option<String>,
}

impl SearchRequest {
    /// Create a new search request with query text and limit.
    pub fn new(query: impl Into<String>, limit: u32) -> Self {
        Self {
            query: query.into(),
            limit: Some(limit),
            db: None,
        }
    }
}

/// Response wrapper from anyrag API.
#[derive(Debug, Deserialize)]
pub struct SearchResponse {
    /// Whether the request was successful.
    pub success: bool,
    /// Search results (present when success is true).
    pub data: Option<Vec<SearchResultItem>>,
    /// Debug information (present when debug=true in query).
    pub debug: Option<serde_json::Value>,
}

/// Single search result from anyrag.
#[derive(Debug, Deserialize, Clone)]
pub struct SearchResultItem {
    /// Document title.
    pub title: String,
    /// Document link/identifier.
    pub link: String,
    /// Document description. May contain token sequence as "tokens:1,2,3,4".
    pub description: String,
    /// Relevance score (cosine similarity for vector search).
    pub score: f64,
}

impl SearchResultItem {
    /// Extract token sequence from description if present.
    ///
    /// Expects format: "tokens:1,2,3,4" somewhere in the description.
    /// Returns `None` if no token sequence found.
    pub fn extract_token_sequence(&self) -> Option<Vec<usize>> {
        let desc = self.description.to_lowercase();
        let marker = "tokens:";
        let start = desc.find(marker)?;
        let rest = &self.description[start + marker.len()..];

        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != ',')
            .unwrap_or(rest.len());
        let token_str = &rest[..end];

        if token_str.is_empty() {
            return None;
        }

        token_str
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .filter(|v: &Vec<usize>| !v.is_empty())
    }
}

// ── Agent Hints (Plan 029, Dynamo Lesson 6) ──────────────────────

/// Per-request hints that signal intent to the speculative decoding pipeline.
///
/// Inspired by NVIDIA Dynamo's `nvext.agent_hints`: a session waiting on user
/// reply has different latency requirements than one running a background tool chain.
///
/// Passed via REST request header (`X-Agent-Hints`) or JSON body, forwarded to
/// `SpeculativeContext` for speculative behavior tuning.
#[derive(Debug, Clone, Default)]
pub struct AgentHints {
    /// Latency sensitivity: 0.0 = background/batch, 1.0 = interactive/real-time.
    /// High sensitivity → more aggressive speculative lookahead, lower draft budget.
    pub latency_sensitivity: f32,
    /// Enable speculative prefill (prompt compression) for this request.
    pub speculative_prefill: bool,
    /// Scheduling priority (0-255). Higher = scheduled sooner.
    pub priority: u8,
}

impl AgentHints {
    /// Create hints for an interactive (user-facing) request.
    pub fn interactive() -> Self {
        Self {
            latency_sensitivity: 1.0,
            speculative_prefill: true,
            priority: 128,
        }
    }

    /// Create hints for a background (batch/tool-chain) request.
    pub fn background() -> Self {
        Self {
            latency_sensitivity: 0.0,
            speculative_prefill: false,
            priority: 0,
        }
    }

    /// Parse hints from a header value string.
    /// Format: `latency=0.8;prefill=true;priority=64`
    /// Unknown keys are ignored. Missing keys get defaults.
    pub fn from_header(value: &str) -> Self {
        let mut hints = Self::default();
        for part in value.split(';') {
            let part = part.trim();
            if let Some((key, val)) = part.split_once('=') {
                match key.trim() {
                    "latency" => {
                        if let Ok(v) = val.trim().parse::<f32>() {
                            hints.latency_sensitivity = v.clamp(0.0, 1.0);
                        }
                    }
                    "prefill" => {
                        hints.speculative_prefill = val.trim().eq_ignore_ascii_case("true");
                    }
                    "priority" => {
                        if let Ok(v) = val.trim().parse::<u8>() {
                            hints.priority = v;
                        }
                    }
                    _ => {} // ignore unknown keys
                }
            }
        }
        hints
    }
}

// ── Tokenize Endpoint (Plan 029, Dynamo Lesson 7) ────────────────

/// Request body for `/v1/tokenize` endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizeRequest {
    /// The text to tokenize.
    pub text: String,
    /// If true, return token strings in addition to IDs.
    #[serde(default)]
    pub include_tokens: bool,
}

/// Response from `/v1/tokenize` endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizeResponse {
    /// Token IDs produced by the tokenizer.
    pub token_ids: Vec<usize>,
    /// Token strings (only present if `include_tokens` was true).
    #[serde(default)]
    pub tokens: Vec<String>,
    /// Total number of tokens.
    pub count: usize,
}

/// Request body for `/v1/detokenize` endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct DetokenizeRequest {
    /// Token IDs to decode back to text.
    pub token_ids: Vec<usize>,
}

/// Response from `/v1/detokenize` endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct DetokenizeResponse {
    /// Decoded text string.
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_request_new() {
        let req = SearchRequest::new("hello world", 5);
        assert_eq!(req.query, "hello world");
        assert_eq!(req.limit, Some(5));
        assert!(req.db.is_none());
    }

    #[test]
    fn test_extract_token_sequence_valid() {
        let result = SearchResultItem {
            title: "test".into(),
            link: "http://example.com".into(),
            description: "some text tokens:1,2,3,4 more text".into(),
            score: 0.95,
        };
        assert_eq!(result.extract_token_sequence(), Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_extract_token_sequence_at_end() {
        let result = SearchResultItem {
            title: "test".into(),
            link: "http://example.com".into(),
            description: "some text tokens:5,6,7".into(),
            score: 0.95,
        };
        assert_eq!(result.extract_token_sequence(), Some(vec![5, 6, 7]));
    }

    #[test]
    fn test_extract_token_sequence_none() {
        let result = SearchResultItem {
            title: "test".into(),
            link: "http://example.com".into(),
            description: "no tokens here".into(),
            score: 0.95,
        };
        assert!(result.extract_token_sequence().is_none());
    }

    #[test]
    fn test_extract_token_sequence_empty_after_marker() {
        let result = SearchResultItem {
            title: "test".into(),
            link: "http://example.com".into(),
            description: "tokens: more text".into(),
            score: 0.95,
        };
        assert!(result.extract_token_sequence().is_none());
    }

    #[test]
    fn test_extract_token_sequence_single_token() {
        let result = SearchResultItem {
            title: "test".into(),
            link: "http://example.com".into(),
            description: "tokens:42".into(),
            score: 0.95,
        };
        assert_eq!(result.extract_token_sequence(), Some(vec![42]));
    }

    // ── AgentHints Tests (Plan 029) ──────────────────────────────

    #[test]
    fn test_agent_hints_default() {
        let hints = AgentHints::default();
        assert_eq!(hints.latency_sensitivity, 0.0);
        assert!(!hints.speculative_prefill);
        assert_eq!(hints.priority, 0);
    }

    #[test]
    fn test_agent_hints_interactive() {
        let hints = AgentHints::interactive();
        assert_eq!(hints.latency_sensitivity, 1.0);
        assert!(hints.speculative_prefill);
        assert!(hints.priority > 0);
    }

    #[test]
    fn test_agent_hints_background() {
        let hints = AgentHints::background();
        assert_eq!(hints.latency_sensitivity, 0.0);
        assert!(!hints.speculative_prefill);
        assert_eq!(hints.priority, 0);
    }

    #[test]
    fn test_agent_hints_from_header_full() {
        let hints = AgentHints::from_header("latency=0.8;prefill=true;priority=64");
        assert!((hints.latency_sensitivity - 0.8).abs() < 1e-6);
        assert!(hints.speculative_prefill);
        assert_eq!(hints.priority, 64);
    }

    #[test]
    fn test_agent_hints_from_header_partial() {
        let hints = AgentHints::from_header("latency=0.5");
        assert!((hints.latency_sensitivity - 0.5).abs() < 1e-6);
        assert!(!hints.speculative_prefill); // default
        assert_eq!(hints.priority, 0); // default
    }

    #[test]
    fn test_agent_hints_from_header_empty() {
        let hints = AgentHints::from_header("");
        assert_eq!(hints.latency_sensitivity, 0.0);
    }

    #[test]
    fn test_agent_hints_from_header_clamps_latency() {
        let hints = AgentHints::from_header("latency=2.0");
        assert!((hints.latency_sensitivity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_agent_hints_from_header_ignores_unknown() {
        let hints = AgentHints::from_header("unknown=foo;latency=0.3");
        assert!((hints.latency_sensitivity - 0.3).abs() < 1e-6);
    }

    // ── Tokenize Types Tests (Plan 029) ──────────────────────────

    #[test]
    fn test_tokenize_request_serialization() {
        let req = TokenizeRequest {
            text: "hello world".into(),
            include_tokens: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("hello world"));
        assert!(json.contains("include_tokens"));
    }

    #[test]
    fn test_tokenize_response_count() {
        let resp = TokenizeResponse {
            token_ids: vec![1, 2, 3],
            tokens: vec!["a".into(), "b".into(), "c".into()],
            count: 3,
        };
        assert_eq!(resp.count, resp.token_ids.len());
    }

    #[test]
    fn test_detokenize_request_serialization() {
        let req = DetokenizeRequest {
            token_ids: vec![1, 2, 3],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("token_ids"));
    }

    #[test]
    fn test_detokenize_response_text() {
        let resp = DetokenizeResponse {
            text: "hello world".into(),
        };
        assert_eq!(resp.text, "hello world");
    }
}
