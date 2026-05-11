//! AI player trait and implementations for Monopoly HL Arena.
//!
//! Four player types representing increasing HL technology levels:
//! - P1 (Random): no model, no learning — pure baseline
//! - P2 (Greedy): heuristic financial scoring
//! - P3 (Validator): heuristic + hard safety validation rules
//! - P4 (Full HL): bandit-adapted strategy with opponent modeling

use std::any::Any;

use super::{
    AUCTION_MIN_BID, BOARD_SIZE, JAIL_FINE, JailDecision, PropertyGroup, SquareKind, TradeOffer,
    TradeResponse, square_kind,
};

// ── Decision Context ───────────────────────────────────────────

/// Read-only snapshot of game state provided to AI for each decision.
///
/// Constructed by the game engine from ECS world state each time a player
/// needs to make a decision. Contains no mutable references — AI reads only.
pub struct DecisionContext {
    /// This player's ID (0–3).
    pub player_id: u8,
    /// Cash on hand.
    pub cash: u32,
    /// Current board position (0–39).
    pub position: u8,
    /// Squares this player owns.
    pub owned_properties: Vec<u8>,
    /// Count of properties owned per color group index (PropertyGroup as usize).
    pub group_counts: [u8; 8],
    /// Cash each opponent holds (index = player_id, 0 if bankrupt).
    pub opponent_cash: [u32; 4],
    /// Number of properties each opponent owns.
    pub opponent_property_count: [u8; 4],
    /// Whether each square is owned by a specific player (None = unowned).
    pub square_owners: [Option<u8>; 40],
    /// Number of houses on each square.
    pub square_houses: [u8; 40],
    /// Whether each square is mortgaged.
    pub square_mortgaged: [bool; 40],
    /// Price printed on the property for each square (0 if not purchasable).
    pub square_prices: [u32; 40],
    /// Rent for 0 houses per square.
    pub square_base_rent: [u32; 40],
    /// House cost per square (0 if not buildable).
    pub square_house_cost: [u32; 40],
    /// Mortgage value per square.
    pub square_mortgage_value: [u32; 40],
    /// Current turn number (1-based).
    pub turn_number: u32,
    /// Whether this player is in jail.
    pub in_jail: bool,
    /// Turns spent in jail so far.
    pub jail_turns: u8,
    /// Whether player holds a Get Out Of Jail Free card.
    pub has_jail_card: bool,
}

impl DecisionContext {
    /// Check if player owns all properties in a color group.
    pub fn owns_complete_set(&self, group: PropertyGroup) -> bool {
        let idx = group as usize;
        self.group_counts[idx] >= group.size()
    }

    /// Count how many properties this player owns in a group.
    pub fn count_in_group(&self, group: PropertyGroup) -> u8 {
        self.group_counts[group as usize]
    }

    /// Squares in a given color group that this player owns.
    pub fn owned_in_group(&self, group: PropertyGroup) -> Vec<u8> {
        let mut result = Vec::new();
        for sq in 0..BOARD_SIZE {
            if matches!(square_kind(sq), SquareKind::Property(g) if g == group)
                && self.square_owners[sq as usize] == Some(self.player_id)
            {
                result.push(sq);
            }
        }
        result
    }

    /// Calculate total net worth approximation (cash + property values).
    pub fn net_worth(&self) -> u32 {
        let mut total = self.cash;
        for &sq in &self.owned_properties {
            let sq_idx = sq as usize;
            if !self.square_mortgaged[sq_idx] {
                total += self.square_prices[sq_idx];
                total += self.square_houses[sq_idx] as u32 * self.square_house_cost[sq_idx];
            } else {
                total += self.square_mortgage_value[sq_idx];
            }
        }
        total
    }

    /// Determine game phase from turn number as primary criterion.
    ///
    /// Turn number drives phase: Early (≤10), Mid (11–25), Late (>25).
    pub fn game_phase(&self) -> GamePhase {
        if self.turn_number <= 10 {
            GamePhase::Early
        } else if self.turn_number <= 25 {
            GamePhase::Mid
        } else {
            GamePhase::Late
        }
    }

    /// Count total houses owned across all properties.
    pub fn total_houses(&self) -> u8 {
        self.owned_properties
            .iter()
            .map(|&sq| self.square_houses[sq as usize])
            .sum()
    }
}

// ── Game Phase ─────────────────────────────────────────────────

/// Macro-level game phase for strategy adaptation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamePhase {
    /// Rounds 1–10 or low property ownership — expand aggressively.
    Early,
    /// Rounds 11–25 or moderate ownership — develop monopolies.
    Mid,
    /// Rounds 26+ or high ownership — survive, optimize cash flow.
    Late,
}

impl GamePhase {
    /// Map each phase to its preferred strategy.
    pub fn preferred_strategy(&self) -> Strategy {
        match self {
            Self::Early => Strategy::Expansion,
            Self::Mid => Strategy::Development,
            Self::Late => Strategy::Survival,
        }
    }
}

// ── Strategy (HL Player) ───────────────────────────────────────

/// Strategy profiles for the HL bandit layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Expansion,
    Development,
    Survival,
    Aggressive,
    Conservative,
}

impl Strategy {
    pub const COUNT: usize = 5;

    pub fn all() -> [Strategy; Self::COUNT] {
        [
            Self::Expansion,
            Self::Development,
            Self::Survival,
            Self::Aggressive,
            Self::Conservative,
        ]
    }

    pub fn as_usize(&self) -> usize {
        match self {
            Self::Expansion => 0,
            Self::Development => 1,
            Self::Survival => 2,
            Self::Aggressive => 3,
            Self::Conservative => 4,
        }
    }

    /// Preferred game phase for each strategy.
    pub fn preferred_phase(&self) -> GamePhase {
        match self {
            Self::Expansion => GamePhase::Early,
            Self::Development => GamePhase::Mid,
            Self::Survival => GamePhase::Late,
            Self::Aggressive => GamePhase::Mid,
            Self::Conservative => GamePhase::Late,
        }
    }
}

// ── Trait ──────────────────────────────────────────────────────

/// AI player trait for Monopoly arena.
///
/// Each implementation represents a different HL technology level:
/// - P1 (Random): no model, no learning
/// - P2 (Greedy): heuristic financial scoring
/// - P3 (Validator): heuristic + safety validation rules
/// - P4 (Full HL): bandit-adapted strategy + opponent modeling
pub trait MonopolyPlayer {
    /// Decide whether to buy a property landed on.
    fn should_buy_property(&mut self, ctx: &DecisionContext, square: u8, price: u32) -> bool;

    /// Bid in an auction. Return 0 to pass.
    fn auction_bid(&mut self, ctx: &DecisionContext, square: u8, current_bid: u32) -> u32;

    /// Decide what to do in jail.
    fn jail_decision(&self, ctx: &DecisionContext) -> JailDecision;

    /// Decide which properties to build houses on. Returns list of square indices.
    fn build_houses(&mut self, ctx: &DecisionContext) -> Vec<u8>;

    /// Respond to a trade offer.
    fn trade_response(&mut self, offer: &TradeOffer, ctx: &DecisionContext) -> TradeResponse;

    /// Propose a trade (called during strategic phase).
    fn propose_trade(&self, ctx: &DecisionContext) -> Option<TradeOffer>;

    /// Priority order of properties to mortgage (first = mortgage first).
    fn mortgage_priority(&self, ctx: &DecisionContext) -> Vec<u8>;

    /// Player display name.
    fn name(&self) -> &str;

    /// Emoji for TUI.
    fn emoji(&self) -> &str;

    /// Reset state for new game.
    fn reset(&mut self);

    /// Downcast support.
    fn as_any(&self) -> &dyn Any;

    /// Downcast support (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ── Helper Functions ───────────────────────────────────────────

/// Calculate strategic value of a property for a player.
///
/// Considers:
/// - Base value from printed price
/// - Bonus for completing a color set (+50%)
/// - Bonus for extending railroad/utility count (+30%)
/// - Penalty for standalone properties (-10%)
pub fn property_strategic_value(ctx: &DecisionContext, square: u8) -> f32 {
    let price = ctx.square_prices[square as usize];
    if price == 0 {
        return 0.0;
    }

    let base_value = price as f32;
    let kind = square_kind(square);

    match kind {
        SquareKind::Property(group) => {
            let owned_in_group = ctx.count_in_group(group);
            let group_size = group.size();

            if ctx.owns_complete_set(group) {
                // Already has monopoly — full value
                base_value * 1.5
            } else if owned_in_group + 1 >= group_size {
                // This property completes the set — big bonus
                base_value * 1.5
            } else if owned_in_group > 0 {
                // Extends existing holdings — moderate bonus
                base_value * (1.0 + 0.2 * owned_in_group as f32 / group_size as f32)
            } else {
                // Standalone — slight discount
                base_value * 0.9
            }
        }
        SquareKind::Railroad => {
            // Count existing railroads owned by this player
            let railroad_squares: [u8; 4] = [5, 15, 25, 35];
            let owned_railroads = railroad_squares
                .iter()
                .filter(|&&sq| ctx.square_owners[sq as usize] == Some(ctx.player_id))
                .count();
            let would_own = owned_railroads
                + if ctx.square_owners[square as usize].is_none() {
                    1
                } else {
                    0
                };

            // Railroads scale exponentially with ownership
            base_value
                * match would_own {
                    1 => 0.6,
                    2 => 1.0,
                    3 => 1.4,
                    4 => 2.0,
                    _ => 0.5,
                }
        }
        SquareKind::Utility => {
            let utility_squares: [u8; 2] = [12, 28];
            let owned_utilities = utility_squares
                .iter()
                .filter(|&&sq| ctx.square_owners[sq as usize] == Some(ctx.player_id))
                .count();

            base_value * if owned_utilities >= 1 { 1.3 } else { 0.8 }
        }
        _ => base_value,
    }
}

/// Check if accepting a trade would give an opponent a monopoly.
///
/// Returns true if any property the opponent would receive completes
/// their color set.
pub fn creates_opponent_monopoly(offer: &TradeOffer, ctx: &DecisionContext) -> bool {
    let check_player = |player_id: u8, properties: &[u8]| -> bool {
        for &sq in properties {
            let kind = square_kind(sq);
            if let SquareKind::Property(group) = kind {
                let mut count = 0u8;
                // Count properties this player already owns in the group
                for board_sq in 0..BOARD_SIZE {
                    if matches!(square_kind(board_sq), SquareKind::Property(g) if g == group)
                        && ctx.square_owners[board_sq as usize] == Some(player_id)
                    {
                        count += 1;
                    }
                }
                // Add the properties from the trade
                for &trade_sq in properties {
                    if matches!(square_kind(trade_sq), SquareKind::Property(g) if g == group) {
                        count += 1;
                    }
                }
                if count >= group.size() {
                    return true;
                }
            }
        }
        false
    };

    // Check if proposer would get a monopoly from responder's properties
    if check_player(offer.proposer, &offer.responder_properties) {
        return true;
    }
    // Check if responder would get a monopoly from proposer's properties
    if check_player(offer.responder, &offer.proposer_properties) {
        return true;
    }

    false
}

/// Calculate the highest rent an opponent could charge.
///
/// Sums max possible rent from opponent's monopolies with houses.
pub fn max_rent_exposure(ctx: &DecisionContext, opponent_id: u8) -> u32 {
    let mut total = 0u32;

    for sq in 0..BOARD_SIZE {
        if ctx.square_owners[sq as usize] != Some(opponent_id) {
            continue;
        }
        if ctx.square_mortgaged[sq as usize] {
            continue;
        }

        let houses = ctx.square_houses[sq as usize];
        let base = ctx.square_base_rent[sq as usize];

        let rent = match houses {
            0 => base,
            1..=4 => base * (houses as u32 * 3),
            _ => base * 15, // hotel
        };

        total += rent;
    }

    total
}

/// Calculate the monopoly bonus multiplier based on property group ownership.
pub fn monopoly_multiplier(ctx: &DecisionContext, group: PropertyGroup) -> f32 {
    if ctx.owns_complete_set(group) {
        2.0
    } else {
        let count = ctx.count_in_group(group);
        let size = group.size();
        1.0 + (count as f32 / size as f32) * 0.5
    }
}

// ── P1: RandomPlayer ───────────────────────────────────────────

/// P1 🎲 — Baseline random player.
///
/// Makes decisions using deterministic pseudo-random derived from square parity
/// and context values. No state, no memory, no model.
pub struct RandomPlayer {
    _id: u8,
}

impl RandomPlayer {
    pub fn new(id: u8) -> Self {
        Self { _id: id }
    }
}

impl MonopolyPlayer for RandomPlayer {
    fn should_buy_property(&mut self, _ctx: &DecisionContext, square: u8, price: u32) -> bool {
        // 50% chance if affordable — use square parity as pseudo-random
        square.is_multiple_of(2) && price <= _ctx.cash
    }

    fn auction_bid(&mut self, _ctx: &DecisionContext, square: u8, current_bid: u32) -> u32 {
        // Random bid: either pass or bid slightly above current
        if current_bid == 0 {
            AUCTION_MIN_BID
        } else if square.is_multiple_of(3) {
            current_bid + AUCTION_MIN_BID
        } else {
            0 // pass
        }
    }

    fn jail_decision(&self, ctx: &DecisionContext) -> JailDecision {
        if ctx.cash >= JAIL_FINE {
            JailDecision::PayFine
        } else {
            JailDecision::RollForDoubles
        }
    }

    fn build_houses(&mut self, _ctx: &DecisionContext) -> Vec<u8> {
        Vec::new() // Random player doesn't strategize about building
    }

    fn trade_response(&mut self, _offer: &TradeOffer, _ctx: &DecisionContext) -> TradeResponse {
        TradeResponse::Decline
    }

    fn propose_trade(&self, _ctx: &DecisionContext) -> Option<TradeOffer> {
        None
    }

    fn mortgage_priority(&self, _ctx: &DecisionContext) -> Vec<u8> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "Random"
    }

    fn emoji(&self) -> &str {
        "🎲"
    }

    fn reset(&mut self) {}

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P2: GreedyPlayer ───────────────────────────────────────────

/// P2 💰 — Heuristic player that maximizes property acquisition.
///
/// Always buys if cash remains above buffer. Builds on highest-rent complete
/// sets first. Accepts trades that increase property count or cash received.
pub struct GreedyPlayer {
    _id: u8,
}

impl GreedyPlayer {
    const CASH_BUFFER: u32 = 100;

    pub fn new(id: u8) -> Self {
        Self { _id: id }
    }
}

impl MonopolyPlayer for GreedyPlayer {
    fn should_buy_property(&mut self, ctx: &DecisionContext, _square: u8, price: u32) -> bool {
        // Buy everything affordable with cash buffer
        ctx.cash > price + Self::CASH_BUFFER
    }

    fn auction_bid(&mut self, ctx: &DecisionContext, square: u8, current_bid: u32) -> u32 {
        // Bid up to max of 90% printed price or 80% strategic value
        let printed_price = ctx.square_prices[square as usize];
        let strategic = property_strategic_value(ctx, square);
        let max_bid = ((printed_price as f32 * 0.9).max(strategic * 0.8) as u32)
            .min(ctx.cash.saturating_sub(Self::CASH_BUFFER));
        let new_bid = current_bid + AUCTION_MIN_BID;

        if new_bid <= max_bid {
            new_bid
        } else {
            0 // pass
        }
    }

    fn jail_decision(&self, ctx: &DecisionContext) -> JailDecision {
        // Pay early (turns 1–15), roll late
        if ctx.turn_number <= 15 && ctx.cash >= JAIL_FINE {
            JailDecision::PayFine
        } else if ctx.has_jail_card {
            JailDecision::UseCard
        } else {
            JailDecision::RollForDoubles
        }
    }

    fn build_houses(&mut self, ctx: &DecisionContext) -> Vec<u8> {
        let mut buildable = Vec::new();

        // Find complete sets and score by rent potential
        for group in PropertyGroup::all() {
            if !ctx.owns_complete_set(group) {
                continue;
            }

            let owned = ctx.owned_in_group(group);
            for sq in owned {
                let sq_idx = sq as usize;
                // Can build if no hotel yet
                if ctx.square_houses[sq_idx] < 5 {
                    let house_cost = ctx.square_house_cost[sq_idx];
                    // Build if can afford (not requiring 2x)
                    if ctx.cash >= house_cost {
                        buildable.push((sq, ctx.square_base_rent[sq_idx]));
                    }
                }
            }
        }

        // Sort by highest rent first
        buildable.sort_by(|a, b| b.1.cmp(&a.1));
        buildable.into_iter().map(|(sq, _)| sq).collect()
    }

    fn trade_response(&mut self, offer: &TradeOffer, ctx: &DecisionContext) -> TradeResponse {
        let our_id = ctx.player_id;

        let (our_properties_given, our_cash_given, their_properties_given, their_cash_given) =
            if offer.proposer == our_id {
                (
                    &offer.proposer_properties,
                    offer.proposer_cash,
                    &offer.responder_properties,
                    offer.responder_cash,
                )
            } else {
                (
                    &offer.responder_properties,
                    offer.responder_cash,
                    &offer.proposer_properties,
                    offer.proposer_cash,
                )
            };

        let net_properties =
            their_properties_given.len() as i32 - our_properties_given.len() as i32;
        let net_cash = their_cash_given as i32 - our_cash_given as i32;

        // Accept if increases property count or cash received
        if net_properties > 0 || net_cash > 0 {
            TradeResponse::Accept
        } else {
            TradeResponse::Decline
        }
    }

    fn propose_trade(&self, ctx: &DecisionContext) -> Option<TradeOffer> {
        // Find a property that would complete our set
        for group in PropertyGroup::all() {
            let owned = ctx.count_in_group(group);
            if owned + 1 != group.size() {
                continue;
            }

            // Find which square in this group we don't own
            for sq in 0..BOARD_SIZE {
                let kind = super::square_kind(sq);
                if let SquareKind::Property(g) = kind {
                    if g != group {
                        continue;
                    }
                    if ctx.square_owners[sq as usize] == Some(ctx.player_id) {
                        continue;
                    }
                    if let Some(owner_id) = ctx.square_owners[sq as usize] {
                        let price = ctx.square_prices[sq as usize];
                        let offer_price = (price as f32 * 1.3) as u32;
                        if ctx.cash > offer_price + Self::CASH_BUFFER {
                            let mut offer = TradeOffer::new(ctx.player_id, owner_id);
                            offer.proposer_cash = offer_price;
                            offer.responder_properties = vec![sq];
                            return Some(offer);
                        }
                    }
                }
            }
        }
        None
    }

    fn mortgage_priority(&self, ctx: &DecisionContext) -> Vec<u8> {
        // Least valuable properties first
        let mut props: Vec<(u8, u32)> = ctx
            .owned_properties
            .iter()
            .filter(|&&sq| !ctx.square_mortgaged[sq as usize])
            .map(|&sq| (sq, ctx.square_prices[sq as usize]))
            .collect();

        props.sort_by_key(|&(_, price)| price);
        props.into_iter().map(|(sq, _)| sq).collect()
    }

    fn name(&self) -> &str {
        "Greedy"
    }

    fn emoji(&self) -> &str {
        "💰"
    }

    fn reset(&mut self) {}

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P3: ValidatorPlayer ────────────────────────────────────────

/// P3 🛡️ — Heuristic + safety validation player.
///
/// All of Greedy's heuristics plus hard safety rules:
/// - Never drops below minimum cash reserve ($200)
/// - Never creates opponent monopoly via trade
/// - Strategic jail decisions based on game phase
/// - Conservative building only when cash flow is healthy
pub struct ValidatorPlayer {
    _id: u8,
    min_cash_reserve: u32,
}

impl ValidatorPlayer {
    const DEFAULT_RESERVE: u32 = 200;
    const BUILD_CASH_THRESHOLD: u32 = 150;
    const BID_SAFETY_MARGIN: f32 = 0.15; // 15% below strategic value

    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            min_cash_reserve: Self::DEFAULT_RESERVE,
        }
    }

    pub fn with_reserve(id: u8, reserve: u32) -> Self {
        Self {
            _id: id,
            min_cash_reserve: reserve,
        }
    }

    /// Calculate maximum safe bid for a property.
    fn max_safe_bid(&self, ctx: &DecisionContext, square: u8) -> u32 {
        let strategic = property_strategic_value(ctx, square);
        let safe_max = ctx.cash.saturating_sub(self.min_cash_reserve);
        let bid_cap = (strategic * (1.0 - Self::BID_SAFETY_MARGIN)) as u32;
        safe_max.min(bid_cap)
    }

    /// Check if a purchase keeps cash above reserve.
    fn can_afford_safely(&self, ctx: &DecisionContext, cost: u32) -> bool {
        ctx.cash > cost + self.min_cash_reserve
    }
}

impl MonopolyPlayer for ValidatorPlayer {
    fn should_buy_property(&mut self, ctx: &DecisionContext, square: u8, price: u32) -> bool {
        if !self.can_afford_safely(ctx, price) {
            return false;
        }
        let kind = super::square_kind(square);
        // Always buy railroads and utilities — safe income, no house investment needed
        if matches!(kind, SquareKind::Railroad | SquareKind::Utility) {
            return true;
        }
        // Always buy if it completes a set
        if let SquareKind::Property(group) = kind {
            let needed = group.size().saturating_sub(ctx.count_in_group(group));
            if needed == 1 {
                return true;
            }
        }
        // Otherwise buy if reasonable value
        let strategic = property_strategic_value(ctx, square);
        strategic >= price as f32 * 0.8
    }

    fn auction_bid(&mut self, ctx: &DecisionContext, square: u8, current_bid: u32) -> u32 {
        let max_bid = self.max_safe_bid(ctx, square);
        let new_bid = current_bid + AUCTION_MIN_BID;

        if new_bid <= max_bid {
            new_bid
        } else {
            0 // pass — won't exceed safety margin
        }
    }

    fn jail_decision(&self, ctx: &DecisionContext) -> JailDecision {
        match ctx.game_phase() {
            // Early game: pay to get out and keep buying
            GamePhase::Early => {
                if ctx.has_jail_card {
                    JailDecision::UseCard
                } else if ctx.cash >= JAIL_FINE + self.min_cash_reserve {
                    JailDecision::PayFine
                } else {
                    JailDecision::RollForDoubles
                }
            }
            // Late game: stay in jail — board is dangerous
            GamePhase::Late => {
                if ctx.jail_turns >= 2 {
                    // Forced out soon anyway
                    if ctx.has_jail_card {
                        JailDecision::UseCard
                    } else {
                        JailDecision::PayFine
                    }
                } else {
                    JailDecision::RollForDoubles // stay safe in jail
                }
            }
            // Mid game: pay if affordable, otherwise roll
            GamePhase::Mid => {
                if ctx.has_jail_card {
                    JailDecision::UseCard
                } else if ctx.cash >= JAIL_FINE + self.min_cash_reserve {
                    JailDecision::PayFine
                } else {
                    JailDecision::RollForDoubles
                }
            }
        }
    }

    fn build_houses(&mut self, ctx: &DecisionContext) -> Vec<u8> {
        // Only build if cash remains >= BUILD_CASH_THRESHOLD
        if ctx.cash < self.min_cash_reserve + Self::BUILD_CASH_THRESHOLD {
            return Vec::new();
        }

        let mut buildable = Vec::new();

        for group in PropertyGroup::all() {
            if !ctx.owns_complete_set(group) {
                continue;
            }

            let owned = ctx.owned_in_group(group);
            for sq in owned {
                let sq_idx = sq as usize;
                if ctx.square_houses[sq_idx] < 5 {
                    let house_cost = ctx.square_house_cost[sq_idx];
                    let remaining = ctx.cash - self.min_cash_reserve;
                    if remaining >= house_cost {
                        let rent_value = ctx.square_base_rent[sq_idx];
                        buildable.push((sq, rent_value, house_cost));
                    }
                }
            }
        }

        // Sort by rent-to-cost ratio (highest value first)
        buildable.sort_by(|a, b| {
            let ratio_a = a.1 as f32 / a.2.max(1) as f32;
            let ratio_b = b.1 as f32 / b.2.max(1) as f32;
            ratio_b
                .partial_cmp(&ratio_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        buildable.into_iter().map(|(sq, _, _)| sq).collect()
    }

    fn trade_response(&mut self, offer: &TradeOffer, ctx: &DecisionContext) -> TradeResponse {
        // First: check if trade creates opponent monopoly — hard block
        if creates_opponent_monopoly(offer, ctx) {
            return TradeResponse::Decline;
        }

        // Then: validate we maintain cash reserve
        let our_cash_given = if offer.proposer == ctx.player_id {
            offer.proposer_cash
        } else {
            offer.responder_cash
        };

        if ctx.cash < our_cash_given + self.min_cash_reserve {
            return TradeResponse::Decline;
        }

        // Finally: use Greedy's heuristic for value
        let our_id = ctx.player_id;
        let (our_properties_given, our_cash_given, their_properties_given, their_cash_given) =
            if offer.proposer == our_id {
                (
                    &offer.proposer_properties,
                    offer.proposer_cash,
                    &offer.responder_properties,
                    offer.responder_cash,
                )
            } else {
                (
                    &offer.responder_properties,
                    offer.responder_cash,
                    &offer.proposer_properties,
                    offer.proposer_cash,
                )
            };

        let net_properties =
            their_properties_given.len() as i32 - our_properties_given.len() as i32;
        let net_cash = their_cash_given as i32 - our_cash_given as i32;

        // More conservative than Greedy: need positive net on both axes
        if net_properties >= 0 && net_cash >= 0 && (net_properties > 0 || net_cash > 0) {
            TradeResponse::Accept
        } else {
            TradeResponse::Decline
        }
    }

    fn propose_trade(&self, ctx: &DecisionContext) -> Option<TradeOffer> {
        // Find a property that would complete our set — with safety margin
        for group in PropertyGroup::all() {
            let owned = ctx.count_in_group(group);
            if owned + 1 != group.size() {
                continue;
            }

            for sq in 0..40u8 {
                let kind = super::square_kind(sq);
                if let SquareKind::Property(g) = kind {
                    if g != group {
                        continue;
                    }
                    if ctx.square_owners[sq as usize] == Some(ctx.player_id) {
                        continue;
                    }
                    if let Some(owner_id) = ctx.square_owners[sq as usize] {
                        let price = ctx.square_prices[sq as usize];
                        // Offer face value — don't overpay like Greedy
                        if ctx.cash > price + self.min_cash_reserve {
                            let mut offer = TradeOffer::new(ctx.player_id, owner_id);
                            offer.proposer_cash = price;
                            offer.responder_properties = vec![sq];
                            return Some(offer);
                        }
                    }
                }
            }
        }
        None
    }

    fn mortgage_priority(&self, ctx: &DecisionContext) -> Vec<u8> {
        // Mortgage least valuable properties first, keep monopoly properties
        let mut props: Vec<(u8, f32)> = ctx
            .owned_properties
            .iter()
            .filter(|&&sq| !ctx.square_mortgaged[sq as usize])
            .map(|&sq| {
                let value = property_strategic_value(ctx, sq);
                // Penalize properties that are part of a complete set
                let kind = square_kind(sq);
                let penalty = match kind {
                    SquareKind::Property(group) if ctx.owns_complete_set(group) => 1000.0,
                    _ => 0.0,
                };
                (sq, value + penalty)
            })
            .collect();

        // Sort by lowest strategic value first (cheapest to lose)
        props.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        props.into_iter().map(|(sq, _)| sq).collect()
    }

    fn name(&self) -> &str {
        "Validator"
    }

    fn emoji(&self) -> &str {
        "\u{1f6e1}\u{fe0f}"
    }

    fn reset(&mut self) {}

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P4: HLPlayer ───────────────────────────────────────────────

/// P4 🧠 — Full HL player with opponent modeling and bandit strategy.
///
/// Combines Validator's safety rules with:
/// - Opponent portfolio tracking across game
/// - Game phase–aware strategy adaptation
/// - Epsilon-greedy bandit for strategy selection
/// - Absorb-compress every 10 games
pub struct HLPlayer {
    _id: u8,
    min_cash_reserve: u32,
    game_count: u32,
    // Opponent tracking
    opponent_properties: Vec<(u8, u8)>, // (square, owner_id)
    // Bandit layer
    strategy_q: [f32; Strategy::COUNT],
    strategy_visits: [u32; Strategy::COUNT],
    pub current_strategy: usize,
    // Compressed arms
    compressed: [bool; Strategy::COUNT],
}

impl HLPlayer {
    const _BANDIT_EPSILON: f32 = 0.1;
    const LEARNING_RATE: f32 = 0.1;

    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            min_cash_reserve: 200,
            game_count: 0,
            opponent_properties: Vec::new(),
            strategy_q: [1.0; Strategy::COUNT], // optimistic init → explores all arms
            strategy_visits: [0; Strategy::COUNT],
            current_strategy: 0, // start with Expansion
            compressed: [false; Strategy::COUNT],
        }
    }

    /// Called at the start of each game. Selects strategy via bandit.
    pub fn start_game(&mut self) {
        self.game_count += 1;
        self.opponent_properties.clear();
        // Pick strategy based on game_count (all games start Early)
        // Use ε-greedy with proper randomness
        let explore = self.game_count.is_multiple_of(10) && !self.compressed.iter().all(|&c| c);

        if explore {
            let available: Vec<usize> = (0..Strategy::COUNT)
                .filter(|&i| !self.compressed[i])
                .collect();
            if !available.is_empty() {
                let pick = available[(self.game_count as usize) % available.len()];
                self.current_strategy = pick;
            }
        } else {
            // Exploit: pick best Q-value strategy
            let mut best_idx = 0;
            let mut best_q = f32::NEG_INFINITY;
            for i in 0..Strategy::COUNT {
                if self.compressed[i] {
                    continue;
                }
                if self.strategy_q[i] > best_q {
                    best_q = self.strategy_q[i];
                    best_idx = i;
                }
            }
            self.current_strategy = best_idx;
        }
    }

    /// Update bandit Q-values after each game based on outcome.
    pub fn update_outcome(&mut self, strategy: Strategy, reward: f32) {
        let idx = strategy.as_usize();
        if self.compressed[idx] {
            return;
        }
        self.strategy_visits[idx] += 1;
        // Q += alpha * (reward - Q)
        self.strategy_q[idx] += Self::LEARNING_RATE * (reward - self.strategy_q[idx]);
    }

    /// Select strategy via epsilon-greedy bandit selection.
    pub fn select_strategy(&mut self, phase: GamePhase) -> Strategy {
        // Phase-adaptive: if current strategy is compressed, re-select
        if self.compressed[self.current_strategy] {
            self.start_game();
        }

        // Optionally adapt mid-game based on phase
        let phase_preferred = phase.preferred_strategy();
        let phase_idx = phase_preferred.as_usize();

        // If phase-appropriate strategy has much higher Q, switch to it
        if !self.compressed[phase_idx]
            && phase_idx != self.current_strategy
            && self.strategy_q[phase_idx] > self.strategy_q[self.current_strategy] + 0.1
        {
            self.current_strategy = phase_idx;
        }

        Strategy::all()[self.current_strategy]
    }

    /// Run absorb-compress cycle. Returns newly compressed strategy indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        let min_visits = 20u32;
        let threshold = 0.1f32;
        let mut newly_compressed = Vec::new();

        for i in 0..Strategy::COUNT {
            if self.compressed[i] {
                continue;
            }
            if self.strategy_visits[i] >= min_visits && self.strategy_q[i] < threshold {
                self.compressed[i] = true;
                newly_compressed.push(i);
            }
        }

        newly_compressed
    }

    /// Generate a compression report string.
    pub fn compress_report(&self) -> String {
        let compressed_count = self.compressed.iter().filter(|&&c| c).count();
        let strategy_names: Vec<String> = Strategy::all()
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let state = if self.compressed[i] { "X" } else { "ok" };
                format!("{:?}({state}:{:.2})", s, self.strategy_q[i])
            })
            .collect();

        format!(
            "Games={} Compressed={}/{} [{}] Strategy={:?}",
            self.game_count,
            compressed_count,
            Strategy::COUNT,
            strategy_names.join(","),
            Strategy::all()[self.current_strategy],
        )
    }

    /// Record an opponent property observation.
    pub fn observe_opponent_property(&mut self, square: u8, owner_id: u8) {
        if let Some(entry) = self
            .opponent_properties
            .iter_mut()
            .find(|(sq, _)| *sq == square)
        {
            entry.1 = owner_id;
        } else {
            self.opponent_properties.push((square, owner_id));
        }
    }

    /// Returns a reference to the bandit Q-value array.
    pub fn strategy_q(&self) -> &[f32; Strategy::COUNT] {
        &self.strategy_q
    }

    /// Returns a reference to the bandit visit count array.
    pub fn strategy_visits(&self) -> &[u32; Strategy::COUNT] {
        &self.strategy_visits
    }

    /// Returns the human-readable strategy names in order.
    pub fn strategy_names() -> [&'static str; Strategy::COUNT] {
        [
            "Expansion",
            "Development",
            "Survival",
            "Aggressive",
            "Conservative",
        ]
    }

    /// Returns total games played by this player.
    pub fn game_count(&self) -> u32 {
        self.game_count
    }

    /// Calculate threat level from an opponent.
    fn threat_level(&self, ctx: &DecisionContext, opponent_id: u8) -> u32 {
        max_rent_exposure(ctx, opponent_id)
    }

    /// Estimate trade value considering both sides' portfolio completion.
    fn evaluate_trade_value(&self, offer: &TradeOffer, ctx: &DecisionContext) -> f32 {
        let our_id = ctx.player_id;

        let (our_props, their_props, their_cash) = if offer.proposer == our_id {
            (
                &offer.proposer_properties,
                &offer.responder_properties,
                offer.responder_cash,
            )
        } else {
            (
                &offer.responder_properties,
                &offer.proposer_properties,
                offer.proposer_cash,
            )
        };

        // Value of properties we receive (considering set completion)
        let mut receive_value = their_cash as f32;
        for &sq in their_props {
            receive_value += property_strategic_value(ctx, sq);
        }

        // Cost of properties we give (considering set disruption)
        let mut give_value = 0.0f32;
        for &sq in our_props {
            give_value += property_strategic_value(ctx, sq) * 1.2; // 20% reluctance penalty
        }

        receive_value - give_value
    }
}

impl MonopolyPlayer for HLPlayer {
    fn should_buy_property(&mut self, ctx: &DecisionContext, square: u8, price: u32) -> bool {
        let strategy = Strategy::all()[self.current_strategy];

        // Base: Validator safety
        if ctx.cash <= price + self.min_cash_reserve {
            return false;
        }

        let strategic_value = property_strategic_value(ctx, square);
        let ratio = if price > 0 {
            strategic_value / price as f32
        } else {
            0.0
        };

        match strategy {
            Strategy::Expansion => ratio > 0.5,
            Strategy::Aggressive => ratio > 0.6,
            Strategy::Development => ratio > 0.8,
            Strategy::Survival => ratio > 1.2,
            Strategy::Conservative => ratio > 1.0,
        }
    }

    fn auction_bid(&mut self, ctx: &DecisionContext, square: u8, current_bid: u32) -> u32 {
        let strategy = Strategy::all()[self.current_strategy];
        let strategic_value = property_strategic_value(ctx, square);

        let max_ratio = match strategy {
            Strategy::Expansion => 0.9,
            Strategy::Aggressive => 0.85,
            Strategy::Development => 0.75,
            Strategy::Survival => 0.6,
            Strategy::Conservative => 0.5,
        };

        let max_bid = ((strategic_value * max_ratio) as u32)
            .min(ctx.cash.saturating_sub(self.min_cash_reserve));

        let new_bid = current_bid + AUCTION_MIN_BID;
        if new_bid <= max_bid { new_bid } else { 0 }
    }

    fn jail_decision(&self, ctx: &DecisionContext) -> JailDecision {
        let phase = ctx.game_phase();

        match phase {
            GamePhase::Early => {
                if ctx.has_jail_card {
                    JailDecision::UseCard
                } else if ctx.cash >= JAIL_FINE + self.min_cash_reserve {
                    JailDecision::PayFine
                } else {
                    JailDecision::RollForDoubles
                }
            }
            GamePhase::Late => {
                // Late: jail is safe — high-rent board is dangerous
                let total_threat: u32 = (0..4u8)
                    .filter(|&id| id != ctx.player_id)
                    .map(|id| self.threat_level(ctx, id))
                    .sum();

                if total_threat > ctx.cash && ctx.jail_turns < 2 {
                    JailDecision::RollForDoubles // stay safe
                } else if ctx.has_jail_card {
                    JailDecision::UseCard
                } else {
                    JailDecision::PayFine
                }
            }
            GamePhase::Mid => {
                if ctx.has_jail_card {
                    JailDecision::UseCard
                } else if ctx.cash >= JAIL_FINE + self.min_cash_reserve {
                    JailDecision::PayFine
                } else {
                    JailDecision::RollForDoubles
                }
            }
        }
    }

    fn build_houses(&mut self, ctx: &DecisionContext) -> Vec<u8> {
        let strategy = Strategy::all()[self.current_strategy];

        // Base safety: need reserve + threshold
        let build_threshold = match strategy {
            Strategy::Aggressive => 100,
            Strategy::Development => 200,
            Strategy::Expansion => 300,
            Strategy::Conservative => 400,
            Strategy::Survival => 500,
        };

        if ctx.cash < self.min_cash_reserve + build_threshold {
            return Vec::new();
        }

        let mut buildable = Vec::new();

        for group in PropertyGroup::all() {
            if !ctx.owns_complete_set(group) {
                continue;
            }

            let owned = ctx.owned_in_group(group);
            for sq in owned {
                let sq_idx = sq as usize;
                if ctx.square_houses[sq_idx] < 5 {
                    let house_cost = ctx.square_house_cost[sq_idx];
                    let remaining = ctx.cash.saturating_sub(self.min_cash_reserve);
                    if remaining >= house_cost {
                        let rent_value = ctx.square_base_rent[sq_idx];
                        // Consider threat level for prioritization
                        let threat_bonus = match strategy {
                            Strategy::Aggressive | Strategy::Development => 1.5,
                            Strategy::Survival => 0.5,
                            _ => 1.0,
                        };
                        buildable.push((sq, (rent_value as f32 * threat_bonus) as u32, house_cost));
                    }
                }
            }
        }

        // Sort by rent potential
        buildable.sort_by(|a, b| b.1.cmp(&a.1));
        buildable.into_iter().map(|(sq, _, _)| sq).collect()
    }

    fn trade_response(&mut self, offer: &TradeOffer, ctx: &DecisionContext) -> TradeResponse {
        // Hard safety: never create opponent monopoly
        if creates_opponent_monopoly(offer, ctx) {
            return TradeResponse::Decline;
        }

        // Cash reserve check
        let our_cash_given = if offer.proposer == ctx.player_id {
            offer.proposer_cash
        } else {
            offer.responder_cash
        };
        if ctx.cash < our_cash_given + self.min_cash_reserve {
            return TradeResponse::Decline;
        }

        // Strategic evaluation
        let trade_value = self.evaluate_trade_value(offer, ctx);

        let strategy = Strategy::all()[self.current_strategy];
        let threshold = match strategy {
            Strategy::Expansion => -50.0,
            Strategy::Aggressive => -20.0,
            Strategy::Development => 0.0,
            Strategy::Conservative => 50.0,
            Strategy::Survival => 100.0,
        };

        if trade_value > threshold {
            TradeResponse::Accept
        } else {
            TradeResponse::Decline
        }
    }

    fn propose_trade(&self, ctx: &DecisionContext) -> Option<TradeOffer> {
        let strategy = Strategy::all()[self.current_strategy];

        // Only Expansion and Aggressive propose trades
        if !matches!(strategy, Strategy::Expansion | Strategy::Aggressive) {
            return None;
        }

        // Find a property that would complete our set
        for group in PropertyGroup::all() {
            let owned = ctx.count_in_group(group);
            let size = group.size();

            // We need exactly 1 more property in this group
            if owned + 1 != size {
                continue;
            }

            // Find the missing property
            let missing: Vec<u8> = (0..BOARD_SIZE)
                .filter(|&sq| {
                    matches!(square_kind(sq), SquareKind::Property(g) if g == group)
                        && ctx.square_owners[sq as usize] != Some(ctx.player_id)
                })
                .collect();

            if let Some(&target_sq) = missing.first()
                && let Some(owner_id) = ctx.square_owners[target_sq as usize]
            {
                if owner_id == ctx.player_id {
                    continue;
                }

                let target_price = ctx.square_prices[target_sq as usize];
                let our_cash_offer = (target_price as f32 * 1.3) as u32; // 30% overpay

                if our_cash_offer <= ctx.cash.saturating_sub(self.min_cash_reserve) {
                    let mut offer = TradeOffer::new(ctx.player_id, owner_id);
                    offer.proposer_cash = our_cash_offer;
                    offer.responder_properties = vec![target_sq];
                    return Some(offer);
                }
            }
        }

        None
    }

    fn mortgage_priority(&self, ctx: &DecisionContext) -> Vec<u8> {
        // Same as Validator but with threat-aware ordering
        let mut props: Vec<(u8, f32)> = ctx
            .owned_properties
            .iter()
            .filter(|&&sq| !ctx.square_mortgaged[sq as usize])
            .map(|&sq| {
                let value = property_strategic_value(ctx, sq);
                let kind = square_kind(sq);

                // Heavy penalty for breaking monopolies
                let penalty = match kind {
                    SquareKind::Property(group) if ctx.owns_complete_set(group) => 5000.0,
                    SquareKind::Property(group) if ctx.count_in_group(group) > 0 => 500.0,
                    SquareKind::Railroad => 200.0,
                    _ => 0.0,
                };
                (sq, value + penalty)
            })
            .collect();

        props.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        props.into_iter().map(|(sq, _)| sq).collect()
    }

    fn name(&self) -> &str {
        "HL"
    }

    fn emoji(&self) -> &str {
        "\u{1f9e0}"
    }

    fn reset(&mut self) {
        self.start_game();

        // Absorb-compress every 10 games
        if self.game_count.is_multiple_of(10) {
            self.compress_cycle();
        }

        // NOTE: Q-values, visits, compressed persist across games (bandit memory)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Factory ────────────────────────────────────────────────────

/// Create the 4 player instances for a tournament.
pub fn create_players() -> Vec<Box<dyn MonopolyPlayer>> {
    vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ]
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a default DecisionContext for testing.
    fn default_ctx() -> DecisionContext {
        DecisionContext {
            player_id: 0,
            cash: 1500,
            position: 0,
            owned_properties: Vec::new(),
            group_counts: [0; 8],
            opponent_cash: [1500; 4],
            opponent_property_count: [0; 4],
            square_owners: [None; 40],
            square_houses: [0; 40],
            square_mortgaged: [false; 40],
            square_prices: [0; 40],
            square_base_rent: [0; 40],
            square_house_cost: [0; 40],
            square_mortgage_value: [0; 40],
            turn_number: 1,
            in_jail: false,
            jail_turns: 0,
            has_jail_card: false,
        }
    }

    /// Create a context with a specific property set up.
    fn ctx_with_property(sq: u8, price: u32, owner: Option<u8>) -> DecisionContext {
        let mut ctx = default_ctx();
        ctx.square_prices[sq as usize] = price;
        ctx.square_base_rent[sq as usize] = price / 10;
        ctx.square_house_cost[sq as usize] = 50;
        ctx.square_mortgage_value[sq as usize] = price / 2;
        if let Some(oid) = owner {
            ctx.square_owners[sq as usize] = Some(oid);
            if oid == ctx.player_id {
                ctx.owned_properties.push(sq);
            }
        }
        ctx
    }

    // ── RandomPlayer Tests ──────────────────────────────

    #[test]
    fn test_random_player_buy_within_bounds() {
        let mut player = RandomPlayer::new(0);
        let mut ctx = default_ctx();
        ctx.cash = 500;

        // Square parity determines buy: even squares can buy, odd cannot
        assert!(player.should_buy_property(&ctx, 2, 100)); // even square, affordable
        assert!(!player.should_buy_property(&ctx, 3, 100)); // odd square
        assert!(!player.should_buy_property(&ctx, 2, 600)); // even square, can't afford
    }

    #[test]
    fn test_random_player_auction_within_bounds() {
        let mut player = RandomPlayer::new(0);
        let ctx = default_ctx();

        // First bid should be AUCTION_MIN_BID
        let bid = player.auction_bid(&ctx, 5, 0);
        assert_eq!(bid, AUCTION_MIN_BID);

        // Subsequent bid: 0 or current + min (depends on square parity)
        let bid2 = player.auction_bid(&ctx, 5, 50);
        assert!(bid2 == 0 || bid2 == 50 + AUCTION_MIN_BID);
    }

    #[test]
    fn test_random_player_jail_pays_if_affordable() {
        let player = RandomPlayer::new(0);
        let ctx = default_ctx();

        let decision = player.jail_decision(&ctx);
        assert_eq!(decision, JailDecision::PayFine); // has $1500 >= $50
    }

    #[test]
    fn test_random_player_jail_rolls_if_broke() {
        let player = RandomPlayer::new(0);
        let mut ctx = default_ctx();
        ctx.cash = 10;

        let decision = player.jail_decision(&ctx);
        assert_eq!(decision, JailDecision::RollForDoubles);
    }

    #[test]
    fn test_random_player_no_build() {
        let mut player = RandomPlayer::new(0);
        let ctx = default_ctx();
        assert!(player.build_houses(&ctx).is_empty());
    }

    #[test]
    fn test_random_player_declines_trades() {
        let mut player = RandomPlayer::new(0);
        let offer = TradeOffer::new(1, 0);
        let ctx = default_ctx();
        assert_eq!(player.trade_response(&offer, &ctx), TradeResponse::Decline);
    }

    // ── GreedyPlayer Tests ──────────────────────────────

    #[test]
    fn test_greedy_player_buys_everything_affordable() {
        let mut player = GreedyPlayer::new(1);
        let ctx = default_ctx();

        // Should buy if cash - price > 50 (buffer)
        assert!(player.should_buy_property(&ctx, 1, 100)); // 1500 - 100 = 1400 > 50
        assert!(player.should_buy_property(&ctx, 1, 500)); // 1500 - 500 = 1000 > 50
        assert!(!player.should_buy_property(&ctx, 1, 1500)); // 1500 - 1500 = 0 < 50
        assert!(!player.should_buy_property(&ctx, 1, 1450)); // 1500 - 1450 = 50, not > 50
    }

    #[test]
    fn test_greedy_player_auction_bids_up_to_90_percent() {
        let mut player = GreedyPlayer::new(1);
        let mut ctx = default_ctx();
        ctx.square_prices[1] = 100; // Mediterranean Ave
        ctx.cash = 1000;

        // Should bid up to 90% of price = 90
        let bid = player.auction_bid(&ctx, 1, 0);
        assert_eq!(bid, AUCTION_MIN_BID); // 10

        let bid2 = player.auction_bid(&ctx, 1, 70);
        assert_eq!(bid2, 70 + AUCTION_MIN_BID); // 80

        let bid3 = player.auction_bid(&ctx, 1, 80);
        assert_eq!(bid3, 80 + AUCTION_MIN_BID); // 90

        let bid4 = player.auction_bid(&ctx, 1, 90);
        assert_eq!(bid4, 0); // pass — above 90% threshold
    }

    #[test]
    fn test_greedy_player_jail_pays_early() {
        let player = GreedyPlayer::new(1);
        let mut ctx = default_ctx();
        ctx.turn_number = 5;

        let decision = player.jail_decision(&ctx);
        assert_eq!(decision, JailDecision::PayFine);
    }

    #[test]
    fn test_greedy_player_jail_rolls_late() {
        let player = GreedyPlayer::new(1);
        let mut ctx = default_ctx();
        ctx.turn_number = 20;
        ctx.cash = 50;

        let decision = player.jail_decision(&ctx);
        assert_eq!(decision, JailDecision::RollForDoubles);
    }

    #[test]
    fn test_greedy_player_mortgage_cheapest_first() {
        let player = GreedyPlayer::new(1);
        let mut ctx = default_ctx();
        ctx.owned_properties = vec![1, 6, 37];
        ctx.square_prices[1] = 60;
        ctx.square_prices[6] = 100;
        ctx.square_prices[37] = 350;

        let priority = player.mortgage_priority(&ctx);
        assert_eq!(priority[0], 1); // cheapest first
        assert_eq!(priority[1], 6);
        assert_eq!(priority[2], 37);
    }

    #[test]
    fn test_greedy_player_trade_accepts_more_properties() {
        let mut player = GreedyPlayer::new(0);
        let ctx = default_ctx();

        // We (player 0) are the responder, receive 2 properties, give 1
        let mut offer = TradeOffer::new(1, 0);
        offer.proposer_properties = vec![6, 8]; // gives us 2
        offer.responder_properties = vec![1]; // we give 1

        let response = player.trade_response(&offer, &ctx);
        assert_eq!(response, TradeResponse::Accept); // net +1 property
    }

    // ── ValidatorPlayer Tests ───────────────────────────

    #[test]
    fn test_validator_never_drops_below_reserve() {
        let mut player = ValidatorPlayer::new(2);
        let mut ctx = default_ctx();
        ctx.square_prices[1] = 100; // needed for strategic value check
        // cash=1500, reserve=200, strategic=100*0.9=90 >= 100*0.8=80
        // 1500 > 100 + 200 => buy
        assert!(player.should_buy_property(&ctx, 1, 100));
        // 1500 - 1300 = 200 = reserve, need strictly > => no buy
        assert!(!player.should_buy_property(&ctx, 1, 1300));
        // 1500 - 1400 = 100 < 200 => no buy
        assert!(!player.should_buy_property(&ctx, 1, 1400));
    }

    #[test]
    fn test_validator_auction_respects_reserve() {
        let mut player = ValidatorPlayer::new(2);
        let mut ctx = default_ctx();
        ctx.cash = 300;
        ctx.square_prices[1] = 100;
        ctx.square_base_rent[1] = 10;

        // Max safe bid = cash - reserve = 300 - 200 = 100
        let bid = player.auction_bid(&ctx, 1, 0);
        assert_eq!(bid, AUCTION_MIN_BID);

        // Should pass if bid would exceed safe max
        let bid2 = player.auction_bid(&ctx, 1, 100);
        assert_eq!(bid2, 0); // 100 + 10 = 110 > 100 safe max
    }

    #[test]
    fn test_validator_jail_stays_late_game() {
        let player = ValidatorPlayer::new(2);
        let mut ctx = default_ctx();
        ctx.turn_number = 30;
        ctx.cash = 1500;

        let decision = player.jail_decision(&ctx);
        assert_eq!(decision, JailDecision::RollForDoubles); // late game = stay safe
    }

    #[test]
    fn test_validator_jail_pays_early_game() {
        let player = ValidatorPlayer::new(2);
        let mut ctx = default_ctx();
        ctx.turn_number = 5;
        ctx.cash = 1500;

        let decision = player.jail_decision(&ctx);
        assert_eq!(decision, JailDecision::PayFine); // early game = pay and get out
    }

    #[test]
    fn test_validator_builds_only_with_sufficient_cash() {
        let mut player = ValidatorPlayer::new(2);
        let mut ctx = default_ctx();
        ctx.cash = 300; // reserve(200) + threshold(200) = 400 > 300 => no build

        // Set up a complete set
        ctx.owned_properties = vec![1, 3]; // Brown set
        ctx.group_counts[PropertyGroup::Brown as usize] = 2;
        ctx.square_houses[1] = 0;
        ctx.square_houses[3] = 0;
        ctx.square_house_cost[1] = 50;
        ctx.square_house_cost[3] = 50;
        ctx.square_base_rent[1] = 2;
        ctx.square_base_rent[3] = 4;
        ctx.square_owners[1] = Some(2);
        ctx.square_owners[3] = Some(2);

        let houses = player.build_houses(&ctx);
        assert!(houses.is_empty()); // Not enough cash above threshold
    }

    #[test]
    fn test_validator_rejects_monopoly_creating_trade() {
        let mut player = ValidatorPlayer::new(0);
        let mut ctx = default_ctx();
        ctx.player_id = 0;

        // Opponent (player 1) owns Baltic Ave (3), trade gives them Mediterranean (1)
        ctx.square_owners[3] = Some(1);
        ctx.opponent_property_count[1] = 1;

        let mut offer = TradeOffer::new(0, 1);
        offer.proposer_properties = vec![1]; // we give Mediterranean
        offer.responder_cash = 100; // opponent gives cash

        let response = player.trade_response(&offer, &ctx);
        assert_eq!(response, TradeResponse::Decline);
    }

    // ── HLPlayer Tests ──────────────────────────────────

    #[test]
    fn test_hl_player_strategy_selection() {
        let mut player = HLPlayer::new(3);

        // start_game selects strategy (game_count becomes 1, exploit picks best Q = all 0 → idx 0)
        player.start_game();
        assert_eq!(player.current_strategy, 0); // Expansion

        // Early game should return Expansion (current strategy)
        let ctx = default_ctx();
        let strategy = player.select_strategy(ctx.game_phase());
        assert_eq!(strategy, Strategy::Expansion);

        // Boost Survival Q-value above optimistic init (1.0) to test mid-game phase adaptation
        player.strategy_q[Strategy::Survival.as_usize()] = 1.5;
        let mut ctx = default_ctx();
        ctx.turn_number = 30; // Late phase prefers Survival
        let strategy = player.select_strategy(ctx.game_phase());
        // Survival Q (1.5) > Expansion Q (1.0) + 0.1, should switch
        assert_eq!(strategy, Strategy::Survival);
    }

    #[test]
    fn test_hl_player_q_value_update() {
        let mut player = HLPlayer::new(3);

        // Optimistic init starts at 1.0 — positive reward keeps it high
        player.update_outcome(Strategy::Expansion, 1.0);
        assert!(player.strategy_q[Strategy::Expansion.as_usize()] >= 1.0);

        // Negative reward pulls Q-value below initial
        let before = player.strategy_q[Strategy::Conservative.as_usize()];
        player.update_outcome(Strategy::Conservative, -1.0);
        assert!(player.strategy_q[Strategy::Conservative.as_usize()] < before);
    }

    #[test]
    fn test_hl_player_respects_reserve() {
        let mut player = HLPlayer::new(3);
        player.current_strategy = Strategy::Expansion.as_usize();
        let mut ctx = default_ctx();
        ctx.cash = 250;

        // Cash = 250, reserve = 200, price = 60 => 250 - 60 = 190 < 200 => no buy
        assert!(!player.should_buy_property(&ctx, 1, 60));
    }

    #[test]
    fn test_hl_player_observes_opponents() {
        let mut player = HLPlayer::new(3);
        player.observe_opponent_property(5, 1);
        player.observe_opponent_property(6, 2);
        player.observe_opponent_property(5, 2); // update railroad to player 2

        assert_eq!(player.opponent_properties.len(), 2);
        assert_eq!(player.opponent_properties[0], (5, 2)); // updated
        assert_eq!(player.opponent_properties[1], (6, 2));
    }

    #[test]
    fn test_hl_player_compression() {
        let mut player = HLPlayer::new(3);

        // Simulate many visits with low Q-value
        for _ in 0..25 {
            player.strategy_visits[Strategy::Conservative.as_usize()] += 1;
        }
        player.strategy_q[Strategy::Conservative.as_usize()] = 0.05;

        let compressed = player.compress_cycle();
        assert!(compressed.contains(&Strategy::Conservative.as_usize()));
        assert!(player.compressed[Strategy::Conservative.as_usize()]);
    }

    #[test]
    fn test_hl_player_propose_trade_completes_set() {
        let mut player = HLPlayer::new(3);
        player.current_strategy = Strategy::Expansion.as_usize();

        let mut ctx = default_ctx();
        ctx.player_id = 0;
        ctx.cash = 2000;

        // Player owns Baltic Ave (3), needs Mediterranean (1) for Brown monopoly
        ctx.owned_properties = vec![3];
        ctx.group_counts[PropertyGroup::Brown as usize] = 1;
        ctx.square_owners[1] = Some(1); // opponent owns Mediterranean
        ctx.square_owners[3] = Some(0); // we own Baltic
        ctx.square_prices[1] = 60;
        ctx.square_prices[3] = 60;

        let trade = player.propose_trade(&ctx);
        assert!(trade.is_some());
        let offer = trade.unwrap();
        assert_eq!(offer.responder_properties, vec![1]); // wants Mediterranean
    }

    // ── Helper Function Tests ───────────────────────────

    #[test]
    fn test_property_strategic_value_standalone() {
        let ctx = ctx_with_property(1, 60, None);
        // Standalone property: 0.9 * price = 54.0
        let value = property_strategic_value(&ctx, 1);
        assert!(value > 0.0);
        assert!(value < 60.0);
    }

    #[test]
    fn test_property_strategic_value_completes_set() {
        let mut ctx = default_ctx();
        ctx.player_id = 0;
        ctx.square_prices[1] = 60;
        ctx.square_prices[3] = 60;
        ctx.square_base_rent[1] = 2;
        ctx.square_base_rent[3] = 4;
        ctx.owned_properties = vec![3];
        ctx.group_counts[PropertyGroup::Brown as usize] = 1;
        ctx.square_owners[3] = Some(0);

        let value = property_strategic_value(&ctx, 1);
        // Completing set: 1.5 * 60 = 90.0
        assert!(value > 60.0);
    }

    #[test]
    fn test_property_strategic_value_railroad() {
        let mut ctx = default_ctx();
        ctx.player_id = 0;
        ctx.square_prices[5] = 200;
        ctx.square_base_rent[5] = 25;
        ctx.square_owners[5] = None;

        // No railroads owned: 0.6 * 200 = 120.0
        let value = property_strategic_value(&ctx, 5);
        assert!((value - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_creates_opponent_monopoly_detection() {
        let mut ctx = default_ctx();
        ctx.player_id = 0;
        ctx.square_owners[3] = Some(1); // opponent owns Baltic

        let mut offer = TradeOffer::new(0, 1);
        offer.proposer_properties = vec![1]; // gives Mediterranean to opponent

        assert!(creates_opponent_monopoly(&offer, &ctx));
    }

    #[test]
    fn test_no_false_monopoly_detection() {
        let ctx = default_ctx();

        let mut offer = TradeOffer::new(0, 1);
        offer.proposer_properties = vec![1]; // gives Mediterranean

        assert!(!creates_opponent_monopoly(&offer, &ctx));
    }

    #[test]
    fn test_max_rent_exposure() {
        let mut ctx = default_ctx();
        ctx.square_owners[6] = Some(1);
        ctx.square_base_rent[6] = 10;
        ctx.square_houses[6] = 2;

        let exposure = max_rent_exposure(&ctx, 1);
        // 2 houses: 10 * (2 * 3) = 60
        assert_eq!(exposure, 60);
    }

    #[test]
    fn test_max_rent_exposure_ignores_mortgaged() {
        let mut ctx = default_ctx();
        ctx.square_owners[6] = Some(1);
        ctx.square_base_rent[6] = 10;
        ctx.square_houses[6] = 2;
        ctx.square_mortgaged[6] = true;

        let exposure = max_rent_exposure(&ctx, 1);
        assert_eq!(exposure, 0);
    }

    // ── Factory Test ────────────────────────────────────

    #[test]
    fn test_create_players() {
        let players = create_players();
        assert_eq!(players.len(), 4);
        assert_eq!(players[0].name(), "Random");
        assert_eq!(players[1].name(), "Greedy");
        assert_eq!(players[2].name(), "Validator");
        assert_eq!(players[3].name(), "HL");
    }

    #[test]
    fn test_create_players_emojis() {
        let players = create_players();
        assert_eq!(players[0].emoji(), "\u{1f3b2}"); // 🎲
        assert_eq!(players[1].emoji(), "\u{1f4b0}"); // 💰
        // Validator and HL use multi-codepoint or different emoji
        assert!(!players[2].emoji().is_empty());
        assert!(!players[3].emoji().is_empty());
    }

    #[test]
    fn test_decision_context_game_phase() {
        let ctx = default_ctx();
        assert_eq!(ctx.game_phase(), GamePhase::Early);

        let mut ctx_mid = default_ctx();
        ctx_mid.turn_number = 15;
        assert_eq!(ctx_mid.game_phase(), GamePhase::Mid);

        let mut ctx_late = default_ctx();
        ctx_late.turn_number = 30;
        assert_eq!(ctx_late.game_phase(), GamePhase::Late);
    }

    #[test]
    fn test_decision_context_owns_complete_set() {
        let mut ctx = default_ctx();
        ctx.group_counts[PropertyGroup::Brown as usize] = 1;
        assert!(!ctx.owns_complete_set(PropertyGroup::Brown)); // needs 2

        ctx.group_counts[PropertyGroup::Brown as usize] = 2;
        assert!(ctx.owns_complete_set(PropertyGroup::Brown));
    }

    #[test]
    fn test_strategy_preferred_phase() {
        assert_eq!(Strategy::Expansion.preferred_phase(), GamePhase::Early);
        assert_eq!(Strategy::Development.preferred_phase(), GamePhase::Mid);
        assert_eq!(Strategy::Survival.preferred_phase(), GamePhase::Late);
    }

    #[test]
    fn test_strategy_roundtrip() {
        for s in Strategy::all() {
            assert_eq!(Strategy::all()[s.as_usize()], s);
        }
    }

    #[test]
    fn test_hl_compress_report() {
        let player = HLPlayer::new(3);
        let report = player.compress_report();
        assert!(report.contains("Games=0"));
        assert!(report.contains("Compressed=0/5"));
    }
}
