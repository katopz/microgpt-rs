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
}
