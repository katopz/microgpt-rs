//! REST client for anyrag vector search.
//!
//! Queries anyrag's `/search/vector` endpoint with text derived from
//! hidden state analysis, then parses token continuations from results.

use crate::rest::types::{SearchRequest, SearchResponse};

/// REST client for anyrag vector search.
pub struct RestClient {
    base_url: String,
    client: reqwest::Client,
}

/// Result from retrieval query.
#[derive(Default)]
pub struct RetrievalResult {
    /// Retrieved token sequences from document metadata.
    pub token_sequences: Vec<Vec<usize>>,
    /// Similarity scores for each sequence.
    pub scores: Vec<f32>,
}

impl RetrievalResult {
    /// Check if retrieval returned any sequences.
    pub fn is_empty(&self) -> bool {
        self.token_sequences.is_empty()
    }
}

/// Errors from REST operations.
#[derive(Debug)]
pub enum RestError {
    /// HTTP request failed.
    Http(reqwest::Error),
    /// JSON parsing failed.
    Json(serde_json::Error),
    /// API returned unsuccessful response.
    Api(String),
}

impl std::fmt::Display for RestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RestError::Http(e) => write!(f, "HTTP error: {e}"),
            RestError::Json(e) => write!(f, "JSON error: {e}"),
            RestError::Api(msg) => write!(f, "API error: {msg}"),
        }
    }
}

impl std::error::Error for RestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RestError::Http(e) => Some(e),
            RestError::Json(e) => Some(e),
            RestError::Api(_) => None,
        }
    }
}

impl From<reqwest::Error> for RestError {
    fn from(e: reqwest::Error) -> Self {
        RestError::Http(e)
    }
}

impl From<serde_json::Error> for RestError {
    fn from(e: serde_json::Error) -> Self {
        RestError::Json(e)
    }
}

/// Convert a hidden state embedding to a summary text for search.
/// Uses simple statistics to create a queryable representation.
fn embedding_to_query(embedding: &[f32], max_tokens: usize) -> String {
    if embedding.is_empty() {
        return String::from("embedding");
    }

    let sum: f32 = embedding.iter().sum();
    let mean = sum / embedding.len() as f32;
    let variance: f32 =
        embedding.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
    let std_dev = variance.sqrt();

    // Quantize embedding into discrete tokens for search
    let tokens: Vec<String> = embedding
        .iter()
        .take(max_tokens)
        .map(|&v| {
            let bucket = ((v - mean) / std_dev.max(1e-6) * 3.0).round() as i32;
            format!("h{bucket}")
        })
        .collect();

    tokens.join(" ")
}

impl RestClient {
    /// Create a new REST client pointing to anyrag base URL.
    ///
    /// Example: `RestClient::new("http://localhost:9090")`
    pub fn new(base_url: &str) -> Self {
        let url = base_url.trim_end_matches('/').to_string();
        Self {
            base_url: url,
            client: reqwest::Client::new(),
        }
    }

    /// Query anyrag /search/vector with hidden state embedding.
    ///
    /// Converts the embedding to a text query, sends to anyrag,
    /// then parses token sequences from search result descriptions.
    ///
    /// Returns historical token continuations ranked by similarity.
    pub async fn retrieve(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<RetrievalResult, RestError> {
        let query = embedding_to_query(embedding, 16);
        self.retrieve_by_query(&query, top_k).await
    }

    /// Query anyrag /search/vector with an explicit text query.
    ///
    /// Useful when you already have the search text prepared.
    pub async fn retrieve_by_query(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<RetrievalResult, RestError> {
        let request = SearchRequest::new(query, top_k as u32);

        let url = format!("{}/search/vector", self.base_url);
        let response = self.client.post(&url).json(&request).send().await?;

        let search_response: SearchResponse = response.json().await?;

        if !search_response.success {
            return Err(RestError::Api("unsuccessful response".into()));
        }

        let results = search_response.data.unwrap_or_default();

        let mut token_sequences = Vec::new();
        let mut scores = Vec::new();

        for result in &results {
            if let Some(tokens) = result.extract_token_sequence() {
                token_sequences.push(tokens);
                scores.push(result.score as f32);
            }
        }

        Ok(RetrievalResult {
            token_sequences,
            scores,
        })
    }

    /// Get the base URL this client is configured with.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_to_query_non_empty() {
        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let query = embedding_to_query(&embedding, 3);
        assert!(!query.is_empty());
        // Should produce quantized tokens
        assert!(query.contains('h'));
    }

    #[test]
    fn test_embedding_to_query_empty() {
        let query = embedding_to_query(&[], 10);
        assert_eq!(query, "embedding");
    }

    #[test]
    fn test_retrieval_result_default() {
        let result = RetrievalResult::default();
        assert!(result.is_empty());
        assert!(result.token_sequences.is_empty());
        assert!(result.scores.is_empty());
    }

    #[test]
    fn test_retrieval_result_not_empty() {
        let result = RetrievalResult {
            token_sequences: vec![vec![1, 2, 3]],
            scores: vec![0.9],
        };
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rest_client_new_trims_trailing_slash() {
        let client = RestClient::new("http://localhost:9090/");
        assert_eq!(client.base_url(), "http://localhost:9090");
    }

    #[test]
    fn test_rest_error_display() {
        let err = RestError::Api("test error".into());
        assert_eq!(format!("{err}"), "API error: test error");
    }
}
