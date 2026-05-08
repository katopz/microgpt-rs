// DataLoader for GPU training: JSONL loading, batching, shuffling.
// Loads training samples from JSONL files (produced by plan 007's exporter).
// Handles next-token prediction: input_ids = tokens[:-1], target_ids = tokens[1:].

use std::io::BufRead;
use std::path::Path;

use serde::Deserialize;

use crate::types::Rng;

// ── Training sample ────────────────────────────────────────────────

/// Single training sample loaded from JSONL.
/// Expects JSON lines with a "tokens" field: `{"tokens": [1, 2, 3, ...]}`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingSample {
    pub tokens: Vec<usize>,
}

// ── DataLoader ──────────────────────────────────────────────────────

/// Batches training samples for GPU consumption.
/// Handles shuffling, padding, and sequence length truncation.
pub struct DataLoader {
    samples: Vec<TrainingSample>,
    batch_size: usize,
    seq_len: usize,
    pad_id: usize,
    rng: Rng,
}

/// Errors during dataloader operations.
#[derive(Debug)]
pub enum DataLoaderError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NoSamples,
}

impl std::fmt::Display for DataLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataLoaderError::Io(e) => write!(f, "IO error: {e}"),
            DataLoaderError::Json(e) => write!(f, "JSON error: {e}"),
            DataLoaderError::NoSamples => write!(f, "No training samples found"),
        }
    }
}

impl std::error::Error for DataLoaderError {}

impl From<std::io::Error> for DataLoaderError {
    fn from(e: std::io::Error) -> Self {
        DataLoaderError::Io(e)
    }
}

impl From<serde_json::Error> for DataLoaderError {
    fn from(e: serde_json::Error) -> Self {
        DataLoaderError::Json(e)
    }
}

impl DataLoader {
    /// Create dataloader from a JSONL file.
    /// Each line should be a JSON object with a "tokens" array.
    pub fn from_jsonl(
        path: &Path,
        batch_size: usize,
        seq_len: usize,
        pad_id: usize,
    ) -> Result<Self, DataLoaderError> {
        let file = std::fs::File::open(path)?;
        let samples: Vec<TrainingSample> = std::io::BufReader::new(file)
            .lines()
            .filter_map(|line: std::io::Result<String>| {
                let line = line.ok()?;
                serde_json::from_str(&line).ok()
            })
            .collect();

        if samples.is_empty() {
            return Err(DataLoaderError::NoSamples);
        }

        Ok(Self {
            samples,
            batch_size,
            seq_len,
            pad_id,
            rng: Rng::new(42),
        })
    }

    /// Create dataloader from in-memory samples (for testing).
    pub fn from_samples(
        samples: Vec<TrainingSample>,
        batch_size: usize,
        seq_len: usize,
        pad_id: usize,
    ) -> Result<Self, DataLoaderError> {
        if samples.is_empty() {
            return Err(DataLoaderError::NoSamples);
        }

        Ok(Self {
            samples,
            batch_size,
            seq_len,
            pad_id,
            rng: Rng::new(42),
        })
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Number of complete batches.
    pub fn num_batches(&self) -> usize {
        self.samples.len() / self.batch_size
    }

    /// Shuffle samples in-place using Fisher-Yates.
    fn shuffle(&mut self) {
        let n = self.samples.len();
        if n <= 1 {
            return;
        }
        for i in (1..n).rev() {
            let j = (self.rng.next() as usize) % (i + 1);
            self.samples.swap(i, j);
        }
    }

    /// Iterate over batches. Shuffles samples first.
    /// Each batch is (input_ids, target_ids) for next-token prediction.
    /// input_ids:  [batch_size * seq_len] — tokens[0..seq_len]
    /// target_ids: [batch_size * seq_len] — tokens[1..seq_len+1]
    pub fn batches(&mut self) -> Vec<(Vec<u32>, Vec<u32>)> {
        self.shuffle();

        self.samples
            .chunks(self.batch_size)
            .filter_map(|batch| {
                // Skip batches that are too small
                if batch.is_empty() {
                    return None;
                }

                let batch_len = batch.len();
                let mut input_ids = Vec::with_capacity(batch_len * self.seq_len);
                let mut target_ids = Vec::with_capacity(batch_len * self.seq_len);

                for sample in batch {
                    let tokens = &sample.tokens;

                    // Need at least 2 tokens for next-token prediction
                    if tokens.len() < 2 {
                        // Fill with padding
                        for _ in 0..self.seq_len {
                            input_ids.push(self.pad_id as u32);
                            target_ids.push(self.pad_id as u32);
                        }
                        continue;
                    }

                    // Build input/target pairs: input[t] = tokens[t], target[t] = tokens[t+1]
                    let usable = (tokens.len() - 1).min(self.seq_len);
                    for t in 0..usable {
                        input_ids.push(tokens[t] as u32);
                        target_ids.push(tokens[t + 1] as u32);
                    }

                    // Pad remaining positions
                    for _ in usable..self.seq_len {
                        input_ids.push(self.pad_id as u32);
                        target_ids.push(self.pad_id as u32);
                    }
                }

                Some((input_ids, target_ids))
            })
            .collect()
    }

    /// Get a single batch for a specific range (for deterministic testing).
    /// No shuffling — returns samples in order.
    pub fn get_batch(&self, start: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        let batch = self.samples.get(start..start + self.batch_size)?;
        let mut input_ids = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut target_ids = Vec::with_capacity(self.batch_size * self.seq_len);

        for sample in batch {
            let tokens = &sample.tokens;
            if tokens.len() < 2 {
                for _ in 0..self.seq_len {
                    input_ids.push(self.pad_id as u32);
                    target_ids.push(self.pad_id as u32);
                }
                continue;
            }

            let usable = (tokens.len() - 1).min(self.seq_len);
            for t in 0..usable {
                input_ids.push(tokens[t] as u32);
                target_ids.push(tokens[t + 1] as u32);
            }
            for _ in usable..self.seq_len {
                input_ids.push(self.pad_id as u32);
                target_ids.push(self.pad_id as u32);
            }
        }

        Some((input_ids, target_ids))
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_dataloader_from_samples() {
        let samples = vec![
            TrainingSample {
                tokens: vec![1, 2, 3, 4, 5],
            },
            TrainingSample {
                tokens: vec![6, 7, 8],
            },
            TrainingSample {
                tokens: vec![10, 11, 12, 13],
            },
        ];

        let mut dl = DataLoader::from_samples(samples, 2, 4, 0).expect("create dataloader");
        assert_eq!(dl.len(), 3);
        assert_eq!(dl.num_batches(), 1);
    }

    #[test]
    fn test_batches_next_token_prediction() {
        let samples = vec![
            TrainingSample {
                tokens: vec![1, 2, 3, 4, 5],
            },
            TrainingSample {
                tokens: vec![6, 7, 8, 9, 10],
            },
        ];

        let mut dl = DataLoader::from_samples(samples, 2, 4, 0).expect("create dataloader");

        // Fix seed for deterministic test
        dl.rng = Rng::new(0);

        let batches = dl.batches();
        assert_eq!(batches.len(), 1);

        let (input_ids, target_ids) = &batches[0];

        // batch_size=2, seq_len=4 → 8 elements each
        assert_eq!(input_ids.len(), 8);
        assert_eq!(target_ids.len(), 8);

        // Check that target[i] = next token after input[i]
        // (within each sample's range, before padding)
    }

    #[test]
    fn test_batch_padding() {
        let samples = vec![TrainingSample {
            tokens: vec![1, 2], // only 1 pair, need 4 → pad 3
        }];

        let mut dl = DataLoader::from_samples(samples, 1, 4, 99).expect("create dataloader");

        let batches = dl.batches();
        assert_eq!(batches.len(), 1);

        let (input_ids, target_ids) = &batches[0];
        assert_eq!(input_ids.len(), 4);
        assert_eq!(target_ids.len(), 4);

        // First position: input=1, target=2
        assert_eq!(input_ids[0], 1);
        assert_eq!(target_ids[0], 2);

        // Remaining: padding
        assert_eq!(input_ids[1], 99);
        assert_eq!(target_ids[1], 99);
        assert_eq!(input_ids[2], 99);
        assert_eq!(input_ids[3], 99);
    }

    #[test]
    fn test_get_batch_deterministic() {
        let samples = vec![
            TrainingSample {
                tokens: vec![1, 2, 3],
            },
            TrainingSample {
                tokens: vec![4, 5, 6],
            },
        ];

        let dl = DataLoader::from_samples(samples, 2, 2, 0).expect("create dataloader");

        let (input_ids, target_ids) = dl.get_batch(0).expect("batch");

        // First sample: [1,2,3] → input=[1,2], target=[2,3]
        assert_eq!(input_ids[0], 1);
        assert_eq!(target_ids[0], 2);
        assert_eq!(input_ids[1], 2);
        assert_eq!(target_ids[1], 3);

        // Second sample: [4,5,6] → input=[4,5], target=[5,6]
        assert_eq!(input_ids[2], 4);
        assert_eq!(target_ids[2], 5);
        assert_eq!(input_ids[3], 5);
        assert_eq!(target_ids[3], 6);
    }

    #[test]
    fn test_empty_samples_error() {
        let result = DataLoader::from_samples(vec![], 1, 4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_jsonl_file() {
        let dir = std::env::temp_dir().join("microgpt_test_dataloader");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.jsonl");

        let mut file = std::fs::File::create(&path).expect("create file");
        writeln!(file, r#"{{"tokens": [1, 2, 3, 4]}}"#).expect("write");
        writeln!(file, r#"{{"tokens": [5, 6, 7]}}"#).expect("write");

        let mut dl = DataLoader::from_jsonl(&path, 2, 3, 0).expect("load jsonl");
        assert_eq!(dl.len(), 2);

        let batches = dl.batches();
        assert_eq!(batches.len(), 1);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_single_token_sample() {
        let samples = vec![TrainingSample {
            tokens: vec![42], // only 1 token → no prediction pair → all padding
        }];

        let mut dl = DataLoader::from_samples(samples, 1, 4, 0).expect("create dataloader");

        let batches = dl.batches();
        assert_eq!(batches.len(), 1);

        let (input_ids, _) = &batches[0];
        // All padding since can't make a prediction pair
        assert!(input_ids.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_truncation_to_seq_len() {
        let samples = vec![TrainingSample {
            tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // 10 tokens
        }];

        let mut dl = DataLoader::from_samples(samples, 1, 3, 0).expect("create dataloader");

        let batches = dl.batches();
        let (input_ids, target_ids) = &batches[0];

        // seq_len=3 → only first 3 positions
        assert_eq!(input_ids.len(), 3);
        assert_eq!(target_ids.len(), 3);

        assert_eq!(input_ids[0], 1);
        assert_eq!(target_ids[0], 2);
        assert_eq!(input_ids[1], 2);
        assert_eq!(target_ids[1], 3);
        assert_eq!(input_ids[2], 3);
        assert_eq!(target_ids[2], 4);
    }
}
