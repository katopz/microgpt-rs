//! Modelless query-hint proposer — template + bandit-driven generation.
//!
//! Rule-based query-hint generator with no neural model, no LoRA, no gradient
//! updates. Uses templates with UCB1-style bandit selection and TrialLog
//! patterns to generate `(query, hint)` pairs targeting the Generator's
//! blind spots.
//!
//! # Why template-based works
//!
//! G-Zero paper's Proposer prompt (Appendix A) is essentially a template
//! with category sampling. Our bandit + TrialLog already tracks which
//! categories have blind spots. TemplateProposer targets those blind spots
//! without needing a neural model — 0 GPU cost, instant generation,
//! fully deterministic.
//!
//! # Template Categories
//!
//! Six categories from G-Zero paper Appendix A:
//! - Writing (email, story, essay, pitch, review)
//! - Explanation (engineer, student, executive)
//! - Advice (career, travel, project)
//! - Analysis (argument, text, product)
//! - Coding (function, debug, design)
//! - Reasoning (logic, math) — capped at ≤1/6 of output
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut proposer = TemplateProposer::new(fastrand::Rng::new());
//!
//! // Generate a query-hint pair
//! let pair = proposer.propose();
//! println!("Query: {}", pair.query);
//! println!("Hint: {}", pair.hint);
//!
//! // Feed back δ for bandit update
//! proposer.observe_delta(pair.template_id, 0.42);
//! ```

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

// ── Query Template ──────────────────────────────────────────────

/// Task type templates from G-Zero paper Appendix A.
///
/// Each variant carries a list of subtypes that parameterize the generated
/// query-hint pair. The subtype is randomly selected during generation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum QueryTemplate {
    /// Writing tasks: email, story, essay, pitch, review.
    Writing { subtypes: Vec<String> },
    /// Explanation tasks for different audiences.
    Explanation { audiences: Vec<String> },
    /// Advice tasks across different domains.
    Advice { domains: Vec<String> },
    /// Analysis tasks: argument, text, product.
    Analysis { types: Vec<String> },
    /// Coding tasks: function, debug, design.
    Coding { languages: Vec<String> },
    /// Reasoning tasks: logic, math.
    ///
    /// Capped at ≤1/6 of total output (paper heuristic) to prevent
    /// over-representation of verifiable domains.
    Reasoning { difficulty: Vec<String> },
}

impl QueryTemplate {
    /// Human-readable category name.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Writing { .. } => "Writing",
            Self::Explanation { .. } => "Explanation",
            Self::Advice { .. } => "Advice",
            Self::Analysis { .. } => "Analysis",
            Self::Coding { .. } => "Coding",
            Self::Reasoning { .. } => "Reasoning",
        }
    }

    /// Whether this template is in the Reasoning category.
    ///
    /// Reasoning is capped at ≤1/6 of total proposals per the paper's
    /// heuristic to prevent over-representation of verifiable domains.
    pub fn is_reasoning(&self) -> bool {
        matches!(self, Self::Reasoning { .. })
    }

    /// Default templates matching G-Zero paper Appendix A.
    pub fn defaults() -> Vec<Self> {
        vec![
            Self::Writing {
                subtypes: vec!["email", "story", "essay", "pitch", "review"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
            Self::Explanation {
                audiences: vec!["engineer", "student", "executive"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
            Self::Advice {
                domains: vec!["career", "travel", "project"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
            Self::Analysis {
                types: vec!["argument", "text", "product"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
            Self::Coding {
                languages: vec!["function", "debug", "design"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
            Self::Reasoning {
                difficulty: vec!["logic", "math"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
        ]
    }
}

// ── Template Stats ──────────────────────────────────────────────

/// Per-template bandit statistics for UCB1 selection.
#[derive(Clone, Debug, Default)]
struct TemplateStats {
    /// Total accumulated δ from observations.
    total_delta: f32,
    /// Number of δ observations (reward feedback).
    delta_count: usize,
    /// Number of times this template was selected for proposal (UCB1 pulls).
    pulls: usize,
}

impl TemplateStats {
    /// Mean δ (Q-value) for this template.
    ///
    /// Returns 0.0 if never observed (no delta feedback yet).
    /// Uses `delta_count` (reward observations) not `pulls` (proposals).
    fn mean_delta(&self) -> f32 {
        if self.delta_count == 0 {
            return 0.0;
        }
        self.total_delta / self.delta_count as f32
    }

    /// UCB1 score: `Q + sqrt(2 * ln(N) / n)`.
    ///
    /// Returns `f32::MAX` for unvisited templates (maximum exploration bonus).
    fn ucb1_score(&self, total_pulls: usize) -> f32 {
        if self.pulls == 0 {
            return f32::MAX;
        }
        let explore = (2.0 * (total_pulls as f32).ln() / self.pulls as f32).sqrt();
        self.mean_delta() + explore
    }
}

// ── Generated Pair ──────────────────────────────────────────────

/// A generated (query, hint) pair from the template proposer.
///
/// The query is the question to pose to the Generator.
/// The hint is the structural guidance to provide alongside the query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeneratedPair {
    /// The question text for the Generator.
    pub query: String,
    /// The hint text to assist the Generator.
    pub hint: String,
    /// Index of the template used to generate this pair.
    pub template_id: usize,
    /// If targeting a blind spot, the arm index with highest δ history.
    pub blind_spot_target: Option<usize>,
}

// ── Template Proposer ────────────────────────────────────────────

/// Modelless proposer: template + bandit-driven query-hint generation.
///
/// Generates `(query, hint)` pairs using rule-based templates with UCB1
/// bandit selection. Biases toward templates with high historical δ,
/// targeting the Generator's blind spots.
///
/// # Architecture
///
/// ```text
/// TemplateProposer
///   ├── templates: Vec<QueryTemplate>     (6 default categories)
///   ├── stats: Vec<TemplateStats>         (per-template δ tracking)
///   ├── reasoning_budget: f32             (≤1/6 cap for Reasoning)
///   ├── rng: fastrand::Rng                (deterministic randomness)
///   └── total_proposals: usize            (for reasoning budget tracking)
/// ```
pub struct TemplateProposer {
    /// Available templates for generation.
    templates: Vec<QueryTemplate>,
    /// Per-template bandit statistics (parallel with `templates`).
    stats: Vec<TemplateStats>,
    /// RNG for subtype selection and randomization.
    rng: fastrand::Rng,
    /// Total proposals generated (for reasoning budget tracking).
    total_proposals: usize,
    /// Maximum fraction of Reasoning proposals (default: 1/6 ≈ 0.167).
    reasoning_cap: f32,
}

impl TemplateProposer {
    /// Create a new template proposer with default templates.
    pub fn new(rng: fastrand::Rng) -> Self {
        let templates = QueryTemplate::defaults();
        let stats = vec![TemplateStats::default(); templates.len()];
        Self {
            templates,
            stats,
            rng,
            total_proposals: 0,
            reasoning_cap: 1.0 / 6.0,
        }
    }

    /// Create with custom templates.
    pub fn with_templates(templates: Vec<QueryTemplate>, rng: fastrand::Rng) -> Self {
        let stats = vec![TemplateStats::default(); templates.len()];
        Self {
            templates,
            stats,
            rng,
            total_proposals: 0,
            reasoning_cap: 1.0 / 6.0,
        }
    }

    /// Set custom reasoning cap fraction (default: 1/6).
    pub fn with_reasoning_cap(mut self, cap: f32) -> Self {
        self.reasoning_cap = cap.clamp(0.0, 1.0);
        self
    }

    /// Generate a query-hint pair targeting the Generator's blind spots.
    ///
    /// Strategy:
    /// 1. If reasoning budget exceeded → exclude Reasoning templates
    /// 2. Select template via UCB1 bandit weighting
    /// 3. Generate (query, hint) from selected template
    ///
    /// The pair targets the Generator's blind spots by biasing toward
    /// templates with high historical δ.
    pub fn propose(&mut self) -> GeneratedPair {
        let template_id = self.bandit_weighted_template();
        let pair = self.generate_from_template(template_id);

        self.stats[template_id].pulls += 1;
        self.total_proposals += 1;

        pair
    }

    /// Generate a query-hint pair targeting a specific blind-spot arm.
    ///
    /// If the arm maps to a known template category, uses that template.
    /// Otherwise falls back to UCB1 selection.
    pub fn propose_targeted(&mut self, blind_spot_arm: usize) -> GeneratedPair {
        let template_id = if blind_spot_arm < self.templates.len() {
            blind_spot_arm
        } else {
            self.bandit_weighted_template()
        };

        let mut pair = self.generate_from_template(template_id);
        pair.blind_spot_target = Some(blind_spot_arm);

        self.stats[template_id].pulls += 1;
        self.total_proposals += 1;

        pair
    }

    /// Feed δ observation for a template — bandit reward signal.
    ///
    /// Updates the UCB1 statistics so future proposals bias toward
    /// templates with high δ (blind-spot-rich categories).
    pub fn observe_delta(&mut self, template_id: usize, delta: f32) {
        if template_id >= self.stats.len() {
            return;
        }
        self.stats[template_id].total_delta += delta.max(0.0);
        self.stats[template_id].delta_count += 1;
    }

    /// Mean δ for a specific template category.
    pub fn mean_delta(&self, template_id: usize) -> f32 {
        self.stats
            .get(template_id)
            .map(|s| s.mean_delta())
            .unwrap_or(0.0)
    }

    /// Number of proposals for a specific template category.
    pub fn pull_count(&self, template_id: usize) -> usize {
        self.stats.get(template_id).map(|s| s.pulls).unwrap_or(0)
    }

    /// Total proposals generated.
    pub fn total_proposals(&self) -> usize {
        self.total_proposals
    }

    /// Number of templates.
    pub fn num_templates(&self) -> usize {
        self.templates.len()
    }

    /// Template at given index.
    pub fn template(&self, id: usize) -> Option<&QueryTemplate> {
        self.templates.get(id)
    }

    /// Whether the reasoning budget is exceeded for this round.
    fn reasoning_budget_exceeded(&self) -> bool {
        if self.total_proposals == 0 {
            return false;
        }
        let reasoning_pulls: usize = self
            .templates
            .iter()
            .zip(self.stats.iter())
            .filter(|(t, _)| t.is_reasoning())
            .map(|(_, s)| s.pulls)
            .sum();
        let fraction = reasoning_pulls as f32 / self.total_proposals as f32;
        fraction > self.reasoning_cap
    }

    /// Select template via UCB1 bandit weighting.
    ///
    /// Excludes Reasoning templates if budget exceeded.
    /// Unvisited templates get maximum exploration bonus.
    fn bandit_weighted_template(&mut self) -> usize {
        let total_pulls: usize = self.stats.iter().map(|s| s.pulls).sum();
        let reasoning_exceeded = self.reasoning_budget_exceeded();

        // Compute UCB1 scores, excluding Reasoning if budget exceeded
        let mut candidates: Vec<(usize, f32)> = self
            .templates
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                // Skip Reasoning if budget exceeded
                !(reasoning_exceeded && t.is_reasoning())
            })
            .map(|(i, _)| (i, self.stats[i].ucb1_score(total_pulls.max(1))))
            .collect();

        // Sort by UCB1 score descending
        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Select from top candidates with some randomness
        let top_k = candidates.len().min(3);
        let idx = self.rng.usize(0..top_k);
        candidates.get(idx).map(|(i, _)| *i).unwrap_or(0)
    }

    /// Generate a (query, hint) pair from a specific template.
    ///
    /// Clones the template to avoid borrow conflict: `self.templates` is immutably
    /// borrowed for lookup, but `self.rng` (inside `pick_subtype`) needs `&mut self`.
    /// Cloning breaks the `self` borrow before the mutable calls.
    fn generate_from_template(&mut self, template_id: usize) -> GeneratedPair {
        // Clone to end the immutable borrow of self.templates before mutable calls
        let Some(template) = self.templates.get(template_id).cloned() else {
            return GeneratedPair {
                query: String::new(),
                hint: String::new(),
                template_id,
                blind_spot_target: None,
            };
        };

        // Extract owned subtype string before calling generate_* (which borrow &mut self)
        let (query, hint) = match template {
            QueryTemplate::Writing { subtypes } => {
                let subtype = self.pick_subtype(&subtypes);
                self.generate_writing(&subtype)
            }
            QueryTemplate::Explanation { audiences } => {
                let audience = self.pick_subtype(&audiences);
                self.generate_explanation(&audience)
            }
            QueryTemplate::Advice { domains } => {
                let domain = self.pick_subtype(&domains);
                self.generate_advice(&domain)
            }
            QueryTemplate::Analysis { types } => {
                let analysis_type = self.pick_subtype(&types);
                self.generate_analysis(&analysis_type)
            }
            QueryTemplate::Coding { languages } => {
                let lang = self.pick_subtype(&languages);
                self.generate_coding(&lang)
            }
            QueryTemplate::Reasoning { difficulty } => {
                let diff = self.pick_subtype(&difficulty);
                self.generate_reasoning(&diff)
            }
        };

        GeneratedPair {
            query,
            hint,
            template_id,
            blind_spot_target: None,
        }
    }

    /// Pick a random subtype from a list.
    fn pick_subtype(&mut self, subtypes: &[String]) -> String {
        if subtypes.is_empty() {
            return String::new();
        }
        subtypes[self.rng.usize(0..subtypes.len())].clone()
    }

    // ── Template Generators ──────────────────────────────────────

    fn generate_writing(&mut self, subtype: &str) -> (String, String) {
        let topics = [
            "a recent breakthrough in renewable energy",
            "the impact of remote work on team dynamics",
            "balancing innovation with stability in engineering",
            "lessons learned from a failed project",
            "the future of human-AI collaboration",
        ];
        let topic = topics[self.rng.usize(0..topics.len())];

        let query = format!("Write a {subtype} about {topic}.");
        let hint = match subtype {
            "email" => "Consider the recipient's perspective. Structure with: context, request, next steps. Tone: professional but approachable.".to_string(),
            "story" => format!(
                "Start with a hook. Use the 'show don't tell' technique. Include a turning point related to {topic}."
            ),
            "essay" => "Open with a thesis statement. Support with 2-3 key arguments. Address a counterpoint. Conclude by connecting back to the broader theme.".to_string(),
            "pitch" => "Lead with the problem. Present your solution as unique. Quantify the impact. End with a clear call-to-action.".to_string(),
            "review" => "Establish criteria upfront. Provide specific examples. Balance strengths and weaknesses. End with a clear recommendation.".to_string(),
            _ => "Structure your response clearly. Consider the audience and purpose.".to_string(),
        };

        (query, hint)
    }

    fn generate_explanation(&mut self, audience: &str) -> (String, String) {
        let concepts = [
            "how neural networks learn through backpropagation",
            "why Rust's ownership model prevents memory bugs",
            "how distributed consensus algorithms work",
            "what makes quantum computing different from classical computing",
            "how databases handle concurrent transactions",
        ];
        let concept = concepts[self.rng.usize(0..concepts.len())];

        let query = format!("Explain {concept} for a {audience}.");
        let hint = match audience {
            "engineer" => "Use technical terminology precisely. Include a concrete implementation detail or code sketch. Reference related systems the reader would know.".to_string(),
            "student" => "Start with an intuitive analogy. Build up from basics. Use diagrams described in words. Check understanding with a quick thought exercise.".to_string(),
            "executive" => "Lead with business impact. Use 2-3 bullet points for key takeaways. Avoid jargon — translate technical terms to business outcomes. Include a 'why this matters now' hook.".to_string(),
            _ => "Be clear and concise. Use examples to illustrate key points.".to_string(),
        };

        (query, hint)
    }

    fn generate_advice(&mut self, domain: &str) -> (String, String) {
        let scenarios = [
            "deciding between two job offers with different trade-offs",
            "planning a complex migration with tight deadlines",
            "navigating a conflict between team members",
            "choosing between building vs buying a critical component",
            "recovering from a public failure or mistake",
        ];
        let scenario = scenarios[self.rng.usize(0..scenarios.len())];

        let query = format!("Give advice on {scenario} in the context of {domain}.");
        let hint = match domain {
            "career" => "Consider both short-term and 5-year impact. Factor in growth potential, not just compensation. Think about what skills each path builds.".to_string(),
            "travel" => "Balance planning with flexibility. Consider the unexpected benefits of each option. Factor in recovery time and hidden costs.".to_string(),
            "project" => "Evaluate risk vs reward for each option. Consider dependencies and bottlenecks. Think about reversibility — which choice is easier to undo?".to_string(),
            _ => "Consider multiple perspectives. Weigh short-term vs long-term consequences. Identify the irreversible decision points.".to_string(),
        };

        (query, hint)
    }

    fn generate_analysis(&mut self, analysis_type: &str) -> (String, String) {
        let subjects = [
            "the argument that microservices are always better than monoliths",
            "a technical blog post claiming 10x performance improvement",
            "a new framework that promises to replace all existing solutions",
            "the trade-offs of static vs dynamic typing in large codebases",
            "whether TDD actually improves software quality",
        ];
        let subject = subjects[self.rng.usize(0..subjects.len())];

        let query = format!("Analyze the {analysis_type} of {subject}.");
        let hint = match analysis_type {
            "argument" => "Identify the main claim and supporting premises. Check for logical fallacies. Consider what evidence would strengthen or weaken the argument. Note unstated assumptions.".to_string(),
            "text" => "Look for the author's credentials and potential biases. Evaluate the strength of cited evidence. Check if conclusions follow from the data presented.".to_string(),
            "product" => "Evaluate against clear criteria: functionality, usability, performance, cost. Compare with alternatives. Identify the target user and whether it serves them well. Note trade-offs explicitly.".to_string(),
            _ => "Be systematic. Define your evaluation criteria upfront. Support each point with evidence.".to_string(),
        };

        (query, hint)
    }

    fn generate_coding(&mut self, task_type: &str) -> (String, String) {
        let problems = [
            "a rate limiter that handles bursty traffic gracefully",
            "a circuit breaker for distributed service calls",
            "an efficient LRU cache with O(1) operations",
            "a retry mechanism with exponential backoff and jitter",
            "a config loader that validates and hot-reloads changes",
        ];
        let problem = problems[self.rng.usize(0..problems.len())];

        let query = format!("Implement {task_type} for {problem} in Rust.");
        let hint = match task_type {
            "function" => "Define clear input/output types first. Consider edge cases: empty input, overflow, concurrent access. Document invariants. Write the signature before the body.".to_string(),
            "debug" => "Start by reproducing the issue reliably. Check recent changes first. Use binary search to isolate the problematic code. Verify your fix doesn't introduce new issues.".to_string(),
            "design" => "Identify the core abstractions. Separate concerns: data, logic, effects. Design for testability. Consider how requirements might evolve. Start with the minimal viable interface.".to_string(),
            _ => "Write clean, idiomatic Rust. Handle errors properly with Result. Add tests for critical paths.".to_string(),
        };

        (query, hint)
    }

    fn generate_reasoning(&mut self, difficulty: &str) -> (String, String) {
        let puzzles = [
            "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
            "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "You have 8 balls, one is heavier. Using a balance scale, what's the minimum weighings to find it?",
            "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
            "A farmer has a fox, chicken, and grain. River crossing with a boat that holds 2. How?",
        ];
        let puzzle = puzzles[self.rng.usize(0..puzzles.len())];

        let query = format!("Solve this {difficulty} problem: {puzzle}");
        let hint = match difficulty {
            "logic" => "Break into smaller steps. Identify the key constraint. Check if your initial intuition might be wrong — these are designed to trick you. Verify your answer satisfies all conditions.".to_string(),
            "math" => "Identify the variables and their relationships. Check if the rate is constant or varies. Consider whether the problem has a linear or non-linear relationship. Verify with a concrete example.".to_string(),
            _ => "Think step by step. State your assumptions. Check your work.".to_string(),
        };

        (query, hint)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_proposer() -> TemplateProposer {
        TemplateProposer::new(fastrand::Rng::new())
    }

    #[test]
    fn test_propose_returns_non_empty_pair() {
        let mut proposer = make_proposer();
        let pair = proposer.propose();

        assert!(!pair.query.is_empty(), "Query should not be empty");
        assert!(!pair.hint.is_empty(), "Hint should not be empty");
        assert!(pair.template_id < proposer.num_templates());
        assert!(pair.blind_spot_target.is_none());
    }

    #[test]
    fn test_propose_increments_total() {
        let mut proposer = make_proposer();

        assert_eq!(proposer.total_proposals(), 0);
        proposer.propose();
        assert_eq!(proposer.total_proposals(), 1);
        proposer.propose();
        assert_eq!(proposer.total_proposals(), 2);
    }

    #[test]
    fn test_propose_targeted_sets_blind_spot() {
        let mut proposer = make_proposer();
        let pair = proposer.propose_targeted(2);

        assert_eq!(pair.blind_spot_target, Some(2));
    }

    #[test]
    fn test_propose_targeted_out_of_bounds_falls_back() {
        let mut proposer = make_proposer();
        let pair = proposer.propose_targeted(99);

        // Should still generate a valid pair (falls back to UCB1)
        assert!(!pair.query.is_empty());
        assert_eq!(pair.blind_spot_target, Some(99));
    }

    #[test]
    fn test_observe_delta_updates_stats() {
        let mut proposer = make_proposer();

        // No observations yet
        assert!((proposer.mean_delta(0)).abs() < 1e-6);

        // Feed δ
        proposer.observe_delta(0, 0.5);
        proposer.observe_delta(0, 0.3);

        // Mean should reflect observations (pulled once from propose, but delta is separate)
        assert!(proposer.mean_delta(0) > 0.0);
    }

    #[test]
    fn test_observe_delta_negative_clamped() {
        let mut proposer = make_proposer();
        proposer.observe_delta(0, -0.5);

        // Negative δ clamped to 0
        assert!((proposer.mean_delta(0)).abs() < 1e-6);
    }

    #[test]
    fn test_observe_delta_out_of_bounds_noop() {
        let mut proposer = make_proposer();
        proposer.observe_delta(99, 0.5); // Should not panic
    }

    #[test]
    fn test_reasoning_budget_cap() {
        let mut proposer = TemplateProposer::new(fastrand::Rng::new()).with_reasoning_cap(0.1);

        // Generate many proposals
        let mut reasoning_count = 0;
        for _ in 0..100 {
            let pair = proposer.propose();
            if let Some(template) = proposer.template(pair.template_id) {
                if template.is_reasoning() {
                    reasoning_count += 1;
                }
            }
        }

        // Reasoning should be capped
        let fraction = reasoning_count as f32 / 100.0;
        assert!(
            fraction <= 0.3, // Some tolerance for randomness
            "Reasoning fraction {fraction:.2} should be roughly capped at 0.1"
        );
    }

    #[test]
    fn test_all_template_categories_covered() {
        let mut proposer = make_proposer();
        let mut seen_categories = std::collections::HashSet::new();

        // Generate enough proposals to likely hit all categories
        for _ in 0..100 {
            let pair = proposer.propose();
            if let Some(template) = proposer.template(pair.template_id) {
                seen_categories.insert(template.category().to_string());
            }
        }

        assert!(
            seen_categories.len() >= 4,
            "Should cover at least 4 categories, got {seen_categories:?}"
        );
    }

    #[test]
    fn test_writing_template_generates_varied_subtypes() {
        let mut proposer = make_proposer();
        let mut subtypes = std::collections::HashSet::new();

        // Force Writing template (template_id = 0)
        for _ in 0..50 {
            let pair = proposer.propose_targeted(0);
            // Extract subtype from query
            if pair.query.starts_with("Write a") {
                let rest = &pair.query["Write a ".len()..];
                if let Some(space) = rest.find(" about") {
                    subtypes.insert(rest[..space].to_string());
                }
            }
        }

        assert!(
            subtypes.len() >= 2,
            "Should generate varied writing subtypes, got {subtypes:?}"
        );
    }

    #[test]
    fn test_query_template_defaults_count() {
        let defaults = QueryTemplate::defaults();
        assert_eq!(
            defaults.len(),
            6,
            "Should have 6 default template categories"
        );
    }

    #[test]
    fn test_query_template_category_names() {
        let defaults = QueryTemplate::defaults();
        assert_eq!(defaults[0].category(), "Writing");
        assert_eq!(defaults[1].category(), "Explanation");
        assert_eq!(defaults[2].category(), "Advice");
        assert_eq!(defaults[3].category(), "Analysis");
        assert_eq!(defaults[4].category(), "Coding");
        assert_eq!(defaults[5].category(), "Reasoning");
    }

    #[test]
    fn test_query_template_is_reasoning() {
        let defaults = QueryTemplate::defaults();
        assert!(!defaults[0].is_reasoning());
        assert!(defaults[5].is_reasoning());
    }

    #[test]
    fn test_ucb1_prioritizes_unvisited() {
        let mut proposer = make_proposer();

        // Propose from template 0 to mark it as visited (increment pulls)
        for _ in 0..5 {
            let _ = proposer.propose_targeted(0);
        }
        // Feed high δ to template 0
        for _ in 0..10 {
            proposer.observe_delta(0, 1.0);
        }
        // Template 3 never proposed from (never pulled)

        // UCB1 should give unvisited templates max score
        let total_pulls: usize = proposer.stats.iter().map(|s| s.pulls).sum();
        let score_0 = proposer.stats[0].ucb1_score(total_pulls.max(1));
        let score_3 = proposer.stats[3].ucb1_score(total_pulls.max(1));

        // Unvisited template should have infinite (MAX) score
        assert_eq!(score_3, f32::MAX);
        // Visited template has finite score
        assert!(score_0 < f32::MAX);
    }

    #[test]
    fn test_generated_pair_serialization_roundtrip() {
        let pair = GeneratedPair {
            query: "Write a story about AI.".into(),
            hint: "Start with a hook.".into(),
            template_id: 0,
            blind_spot_target: Some(3),
        };

        let json = serde_json::to_string(&pair).unwrap();
        let deserialized: GeneratedPair = serde_json::from_str(&json).unwrap();
        assert_eq!(pair, deserialized);
    }

    #[test]
    fn test_with_custom_templates() {
        let templates = vec![QueryTemplate::Writing {
            subtypes: vec!["haiku".into(), "sonnet".into()],
        }];
        let proposer = TemplateProposer::with_templates(templates, fastrand::Rng::new());

        assert_eq!(proposer.num_templates(), 1);
    }

    #[test]
    fn test_pull_count_tracking() {
        let mut proposer = make_proposer();

        // Force template 0
        let _ = proposer.propose_targeted(0);
        assert_eq!(proposer.pull_count(0), 1);

        let _ = proposer.propose_targeted(0);
        assert_eq!(proposer.pull_count(0), 2);
    }
}
