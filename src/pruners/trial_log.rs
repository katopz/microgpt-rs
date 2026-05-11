//! Trial persistence for Heuristic Learning — JSONL episode history.
//!
//! Records every bandit episode to a JSONL file for offline analysis,
//! absorb-compress decisions, and regression testing.
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut log = TrialLog::new(Path::new("/tmp/trials.jsonl"))?;
//! log.append(&TrialRecord { episode: 0, arm: 2, reward: 1.0, ..Default::default() });
//! log.flush()?;
//!
//! let records = TrialLog::load(Path::new("/tmp/trials.jsonl"))?;
//! let summary = TrialLog::summary(&records);
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Result, Write};
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::review_metrics::{ReviewMetrics, ReviewSummary};

// ── Record ──────────────────────────────────────────────────────

/// A single episode record for persistent trial history.
///
/// Maps 1:1 with the HL article's `trials.jsonl` format.
/// Serialized as one JSON line per record.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrialRecord {
    /// Episode index (0-based).
    pub episode: usize,
    /// Arm (action) selected.
    pub arm: usize,
    /// Observed reward ∈ [0.0, 1.0].
    pub reward: f32,
    /// Q-value estimate at time of selection.
    pub q_value: f32,
    /// Running sum of rewards through this episode.
    pub cumulative_reward: f32,
    /// Running sum of regret through this episode.
    pub cumulative_regret: f32,
    /// Configuration snapshot (strategy, env params, etc.).
    pub config: String,
    /// Free-form annotation (e.g., "compress triggered").
    pub note: String,
    /// Whether the base pruner (without review) was correct (Plan 036).
    pub base_correct: Option<bool>,
    /// Whether the reviewed decision was correct (Plan 036).
    pub reviewed_correct: Option<bool>,
}

// ── Summary ─────────────────────────────────────────────────────

/// Aggregate statistics over a set of trial records.
#[derive(Clone, Debug)]
pub struct TrialSummary {
    /// Total number of episodes.
    pub total_episodes: usize,
    /// Arm with highest average reward.
    pub best_arm: usize,
    /// Average reward across all episodes.
    pub avg_reward: f32,
    /// Average regret across all episodes.
    pub avg_regret: f32,
}

// ── TrialLog ────────────────────────────────────────────────────

/// Append-only JSONL trial log for persistent episode history.
///
/// Each call to [`append`] writes one serialized [`TrialRecord`] as a JSON line.
/// Buffered for throughput — call [`flush`] to ensure durability.
pub struct TrialLog {
    writer: BufWriter<File>,
    path: std::path::PathBuf,
    count: usize,
    /// Optional review metrics for inference-time feedback tracking (Plan 036).
    review_metrics: Option<Arc<ReviewMetrics>>,
}

impl TrialLog {
    /// Create or append to a JSONL trial log at `path`.
    ///
    /// Creates the file if it doesn't exist. Appends if it does.
    pub fn new(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().append(true).create(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
            count: 0,
            review_metrics: None,
        })
    }

    /// Enable review metrics tracking (builder pattern).
    ///
    /// When enabled, `append_with_review` updates both the log and metrics.
    /// The same `Arc<ReviewMetrics>` can be shared across components.
    pub fn with_review_metrics(mut self, metrics: Arc<ReviewMetrics>) -> Self {
        self.review_metrics = Some(metrics);
        self
    }

    /// Append a single trial record as one JSON line.
    ///
    /// Buffered — may not hit disk until [`flush`](Self::flush).
    pub fn append(&mut self, record: &TrialRecord) -> Result<()> {
        let line = serde_json::to_string(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
    }

    /// Path of the JSONL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of records written so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Load all trial records from a JSONL file.
    ///
    /// Malformed lines are skipped with a warning to stderr.
    pub fn load(path: &Path) -> Result<Vec<TrialRecord>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<TrialRecord>(trimmed) {
                Ok(record) => records.push(record),
                Err(e) => eprintln!(
                    "trial_log: skipping malformed line {} in {}: {e}",
                    line_num + 1,
                    path.display()
                ),
            }
        }

        Ok(records)
    }

    /// Append a trial record with review classification.
    ///
    /// Writes the record to the JSONL log AND updates review metrics
    /// (if enabled via `with_review_metrics`).
    pub fn append_with_review(
        &mut self,
        record: &TrialRecord,
        base_correct: bool,
        reviewed_correct: bool,
    ) -> Result<()> {
        // Update review metrics if enabled
        if let Some(ref metrics) = self.review_metrics {
            metrics.record(base_correct, reviewed_correct);
        }
        self.append(record)
    }

    /// Read-only access to review metrics (if enabled).
    pub fn metrics(&self) -> Option<&ReviewMetrics> {
        self.review_metrics.as_ref().map(|arc| arc.as_ref())
    }

    /// Convenience: compute review metrics summary (if enabled).
    pub fn metrics_summary(&self) -> Option<ReviewSummary> {
        self.review_metrics.as_ref().map(|m| m.summary())
    }

    /// Compute aggregate summary over a slice of trial records.
    ///
    /// Returns default summary for empty input.
    pub fn summary(records: &[TrialRecord]) -> TrialSummary {
        if records.is_empty() {
            return TrialSummary {
                total_episodes: 0,
                best_arm: 0,
                avg_reward: 0.0,
                avg_regret: 0.0,
            };
        }

        let total_episodes = records.len();
        let avg_reward = records.iter().map(|r| r.reward).sum::<f32>() / total_episodes as f32;
        let avg_regret = records
            .iter()
            .map(|r| r.cumulative_regret)
            .next_back()
            .unwrap_or(0.0)
            / total_episodes as f32;

        // Find best arm by average reward
        let mut arm_counts: std::collections::HashMap<usize, (f32, usize)> =
            std::collections::HashMap::new();
        for r in records {
            let entry = arm_counts.entry(r.arm).or_insert((0.0, 0));
            entry.0 += r.reward;
            entry.1 += 1;
        }
        let best_arm = arm_counts
            .iter()
            .map(|(&arm, &(sum, count))| (arm, sum / count as f32))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arm, _)| arm)
            .unwrap_or(0);

        TrialSummary {
            total_episodes,
            best_arm,
            avg_reward,
            avg_regret,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "microgpt_test_{name}_{pid}.jsonl",
            pid = std::process::id()
        ))
    }

    fn sample_record(episode: usize, arm: usize, reward: f32) -> TrialRecord {
        TrialRecord {
            episode,
            arm,
            reward,
            q_value: 0.5,
            cumulative_reward: reward * episode as f32,
            cumulative_regret: (1.0 - reward) * episode as f32,
            config: "test".into(),
            note: format!("ep{episode}"),
            base_correct: None,
            reviewed_correct: None,
        }
    }

    #[test]
    fn test_roundtrip() {
        let path = temp_path("roundtrip");
        let _ = std::fs::remove_file(&path);

        // Write 3 records
        {
            let mut log = TrialLog::new(&path).unwrap();
            for i in 0..3 {
                log.append(&sample_record(i, i, 0.5)).unwrap();
            }
            log.flush().unwrap();
        }

        // Load back
        let records = TrialLog::load(&path).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0], sample_record(0, 0, 0.5));
        assert_eq!(records[2], sample_record(2, 2, 0.5));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_summary_aggregation() {
        let records = vec![
            sample_record(0, 0, 0.2),
            sample_record(1, 1, 0.9),
            sample_record(2, 1, 0.9),
            sample_record(3, 0, 0.1),
        ];

        let summary = TrialLog::summary(&records);
        assert_eq!(summary.total_episodes, 4);
        assert_eq!(summary.best_arm, 1); // arm 1 avg = 0.9 > arm 0 avg = 0.15
        assert!((summary.avg_reward - 0.525).abs() < 0.01);
    }

    #[test]
    fn test_empty_log() {
        let path = temp_path("empty");
        let _ = std::fs::remove_file(&path);

        // Write nothing
        {
            let mut log = TrialLog::new(&path).unwrap();
            log.flush().unwrap();
        }

        let records = TrialLog::load(&path).unwrap();
        assert!(records.is_empty());

        let summary = TrialLog::summary(&records);
        assert_eq!(summary.total_episodes, 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_count_tracks_appends() {
        let path = temp_path("count");
        let _ = std::fs::remove_file(&path);

        let mut log = TrialLog::new(&path).unwrap();
        assert_eq!(log.count(), 0);

        log.append(&sample_record(0, 0, 0.5)).unwrap();
        assert_eq!(log.count(), 1);

        log.append(&sample_record(1, 1, 0.8)).unwrap();
        assert_eq!(log.count(), 2);

        let _ = std::fs::remove_file(&path);
    }
}
