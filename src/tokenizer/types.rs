use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// BPE tokenizer with vocabulary and merge rules.
#[derive(Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// Token string → ID mapping.
    #[serde(with = "map_serde")]
    pub vocab_to_id: HashMap<String, usize>,
    /// ID → token string mapping.
    pub id_to_vocab: Vec<String>,
    /// Ordered merge rules: apply in this order during encoding.
    pub merges: Vec<MergeRule>,
    /// Merge rule → rank (index in merges vec) for fast lookup.
    #[serde(skip)]
    pub merge_ranks: HashMap<(String, String), usize>,
    /// Beginning-of-sequence token ID.
    pub bos_id: usize,
    /// End-of-sequence token ID.
    pub eos_id: usize,
    /// Padding token ID.
    pub pad_id: usize,
}

/// A single BPE merge rule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
}

impl BpeTokenizer {
    /// Unknown token ID (last entry in vocabulary).
    pub fn unk_id(&self) -> usize {
        self.id_to_vocab.len().saturating_sub(1)
    }

    /// Rebuild merge_ranks from merges vector.
    pub fn rebuild_ranks(&mut self) {
        self.merge_ranks = self
            .merges
            .iter()
            .enumerate()
            .map(|(rank, rule)| ((rule.left.clone(), rule.right.clone()), rank))
            .collect();
    }
}

mod map_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(map: &HashMap<String, usize>, s: S) -> Result<S::Ok, S::Error> {
        let vec: Vec<(&String, &usize)> = map.iter().collect();
        vec.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<String, usize>, D::Error> {
        let vec: Vec<(String, usize)> = Vec::deserialize(d)?;
        Ok(vec.into_iter().collect())
    }
}
