//! Monopoly Board Game Engine — ECS-based Monopoly with Heuristic Learning AI
//!
//! bevy_ecs standalone engine where 4 AI players compete in Monopoly,
//! using progressively more sophisticated HL strategies.

pub mod board;
pub mod players;
pub mod systems;

pub use board::{build_board, shuffle_decks};
pub use players::{
    DecisionContext, GamePhase, GreedyPlayer, HLPlayer, MonopolyPlayer, RandomPlayer, Strategy,
    ValidatorPlayer,
};
pub use systems::*;

use std::fmt;

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────

pub const BOARD_SIZE: u8 = 40;
pub const STARTING_CASH: u32 = 1500;
pub const GO_SALARY: u32 = 200;
pub const JAIL_FINE: u32 = 50;
pub const JAIL_SQUARE: u8 = 10;
pub const GO_TO_JAIL_SQUARE: u8 = 30;
pub const FREE_PARKING_SQUARE: u8 = 20;
pub const GO_SQUARE: u8 = 0;
pub const INCOME_TAX_SQUARE: u8 = 4;
pub const LUXURY_TAX_SQUARE: u8 = 38;
pub const MAX_JAIL_TURNS: u8 = 3;
pub const MAX_DOUBLES: u8 = 3;
pub const MAX_HOUSES: u8 = 4; // 5 = hotel
pub const AUCTION_MIN_BID: u32 = 10;
pub const MORTGAGE_INTEREST_RATE: f32 = 0.10; // 10%

// ── Enums ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyGroup {
    Brown,
    LightBlue,
    Pink,
    Orange,
    Red,
    Yellow,
    Green,
    DarkBlue,
}

impl PropertyGroup {
    /// Returns all property groups in board order.
    pub fn all() -> [PropertyGroup; 8] {
        [
            Self::Brown,
            Self::LightBlue,
            Self::Pink,
            Self::Orange,
            Self::Red,
            Self::Yellow,
            Self::Green,
            Self::DarkBlue,
        ]
    }

    /// Number of properties in this color group.
    pub fn size(&self) -> u8 {
        match self {
            Self::Brown | Self::DarkBlue => 2,
            Self::LightBlue
            | Self::Pink
            | Self::Orange
            | Self::Red
            | Self::Yellow
            | Self::Green => 3,
        }
    }

    /// House cost for this color group.
    pub fn house_cost(&self) -> u32 {
        match self {
            Self::Brown | Self::LightBlue => 50,
            Self::Pink | Self::Orange => 100,
            Self::Red | Self::Yellow => 150,
            Self::Green | Self::DarkBlue => 200,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SquareKind {
    Go,
    Property(PropertyGroup),
    Railroad,
    Utility,
    Tax(TaxKind),
    Chance,
    CommunityChest,
    Jail,
    FreeParking,
    GoToJail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TaxKind {
    Income,
    Luxury,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnPhase {
    PreTurn,
    Rolling,
    Resolving,
    Acquisition,
    Auction,
    FinancialCrisis,
    Strategic,
    EndTurn,
    Bankrupt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JailReason {
    LandedOnGoToJail,
    Speeding,
    CardEffect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseMethod {
    PaidFine,
    UsedCard,
    RolledDoubles,
    MaxTurnsExceeded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JailDecision {
    PayFine,
    UseCard,
    RollForDoubles,
}

#[derive(Clone, Debug)]
pub enum CardEffect {
    CollectMoney(u32),
    PayMoney(u32),
    PayPerHouse { house: u32, hotel: u32 },
    MoveTo(u8),
    MoveBack(u8),
    MoveToNearest { is_railroad: bool },
    GoToJail,
    GetOutOfJailFree,
    PayEachPlayer(u32),
    CollectFromEachPlayer(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum TradeResponse {
    Accept,
    Decline,
    CounterOffer(TradeOffer),
}

// ── Components ─────────────────────────────────────────────────

#[derive(Component)]
pub struct Player {
    pub id: u8,
    pub cash: u32,
    pub position: u8,
    pub in_jail: bool,
    pub jail_turns: u8,
    pub get_out_of_jail_free: u8,
    pub doubles_count: u8,
    pub is_bankrupt: bool,
}

impl Player {
    pub fn new(id: u8, starting_cash: u32) -> Self {
        Self {
            id,
            cash: starting_cash,
            position: GO_SQUARE,
            in_jail: false,
            jail_turns: 0,
            get_out_of_jail_free: 0,
            doubles_count: 0,
            is_bankrupt: false,
        }
    }

    pub fn pay(&mut self, amount: u32) -> bool {
        if self.cash >= amount {
            self.cash -= amount;
            true
        } else {
            false
        }
    }

    pub fn receive(&mut self, amount: u32) {
        self.cash += amount;
    }

    pub fn net_worth(&self) -> u32 {
        // Note: full net worth would include property values, computed at system level
        self.cash
    }
}

#[derive(Component)]
pub struct Property {
    pub square: u8,
    pub group: PropertyGroup,
    pub name: &'static str,
    pub price: u32,
    pub base_rent: u32,
    pub monopoly_rent: u32,
    pub house_cost: u32,
    pub house_rent: [u32; 5],
    pub mortgage_value: u32,
}

#[derive(Component)]
pub struct Owned {
    pub owner: Entity,
    pub is_mortgaged: bool,
    pub houses: u8, // 0-4 = houses, 5 = hotel
}

impl Owned {
    pub fn new(owner: Entity) -> Self {
        Self {
            owner,
            is_mortgaged: false,
            houses: 0,
        }
    }

    pub fn has_hotel(&self) -> bool {
        self.houses > MAX_HOUSES
    }

    pub fn house_count(&self) -> u8 {
        if self.has_hotel() { 0 } else { self.houses }
    }
}

#[derive(Component)]
pub struct Railroad;

#[derive(Component)]
pub struct Utility;

#[derive(Component)]
pub struct CardDeck {
    pub cards: Vec<CardEffect>,
    pub draw_index: usize,
    pub is_chance: bool,
}

impl CardDeck {
    pub fn draw(&mut self) -> &CardEffect {
        let idx = self.draw_index % self.cards.len();
        self.draw_index = idx + 1;
        &self.cards[idx]
    }

    pub fn shuffle(&mut self, seed: u64) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let n = self.cards.len();
        for i in (1..n).rev() {
            let j = rng.usize(..=i);
            self.cards.swap(i, j);
        }
        self.draw_index = 0;
    }
}

#[derive(Component)]
pub struct BoardSquare {
    pub index: u8,
    pub kind: SquareKind,
}

// ── TradeOffer ─────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct TradeOffer {
    pub proposer: u8,
    pub responder: u8,
    pub proposer_properties: Vec<u8>,
    pub proposer_cash: u32,
    pub responder_properties: Vec<u8>,
    pub responder_cash: u32,
}

impl TradeOffer {
    pub fn new(proposer: u8, responder: u8) -> Self {
        Self {
            proposer,
            responder,
            proposer_properties: Vec::new(),
            proposer_cash: 0,
            responder_properties: Vec::new(),
            responder_cash: 0,
        }
    }

    pub fn net_value(&self) -> i64 {
        let proposer_gives =
            self.proposer_cash as i64 + (self.proposer_properties.len() as i64 * 100); // rough estimate
        let responder_gives =
            self.responder_cash as i64 + (self.responder_properties.len() as i64 * 100);
        responder_gives - proposer_gives
    }
}

// ── Resources ──────────────────────────────────────────────────

#[derive(Resource)]
pub struct Board {
    pub squares: [Entity; BOARD_SIZE as usize],
}

#[derive(Resource)]
pub struct TurnState {
    pub current_player: u8,
    pub phase: TurnPhase,
    pub turn_number: u32,
    pub doubles_count: u8,
}

impl TurnState {
    pub fn new(starting_player: u8) -> Self {
        Self {
            current_player: starting_player,
            phase: TurnPhase::PreTurn,
            turn_number: 1,
            doubles_count: 0,
        }
    }

    pub fn advance_player(&mut self, active_players: u8) {
        self.current_player = (self.current_player + 1) % active_players;
        if self.current_player == 0 {
            self.turn_number += 1;
        }
        self.phase = TurnPhase::PreTurn;
        self.doubles_count = 0;
    }
}

#[derive(Resource)]
pub struct GameConfig {
    pub starting_cash: u32,
    pub salary: u32,
    pub jail_fine: u32,
    pub max_jail_turns: u8,
    pub max_doubles: u8,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            starting_cash: STARTING_CASH,
            salary: GO_SALARY,
            jail_fine: JAIL_FINE,
            max_jail_turns: MAX_JAIL_TURNS,
            max_doubles: MAX_DOUBLES,
        }
    }
}

#[derive(Resource)]
pub struct PlayerEntities {
    pub entities: [Entity; 4],
}

#[derive(Resource, Default)]
pub struct Statistics {
    pub turns_played: u32,
    pub properties_bought: [u32; 4],
    pub rent_paid: [u32; 4],
    pub houses_built: [u32; 4],
    pub trades_completed: [u32; 4],
}

impl Statistics {
    pub fn record_purchase(&mut self, player: u8) {
        if let Some(stat) = self.properties_bought.get_mut(player as usize) {
            *stat += 1;
        }
    }

    pub fn record_rent(&mut self, player: u8, amount: u32) {
        if let Some(stat) = self.rent_paid.get_mut(player as usize) {
            *stat += amount;
        }
    }

    pub fn record_house(&mut self, player: u8) {
        if let Some(stat) = self.houses_built.get_mut(player as usize) {
            *stat += 1;
        }
    }

    pub fn record_trade(&mut self, player: u8) {
        if let Some(stat) = self.trades_completed.get_mut(player as usize) {
            *stat += 1;
        }
    }
}

// ── Events ─────────────────────────────────────────────────────

#[derive(Event, Clone, Debug)]
pub enum GameEvent {
    TurnStarted {
        player: u8,
    },
    DiceRolled {
        player: u8,
        die1: u8,
        die2: u8,
        doubles: bool,
    },
    PlayerMoved {
        player: u8,
        from: u8,
        to: u8,
        passed_go: bool,
    },
    SalaryCollected {
        player: u8,
        amount: u32,
    },
    PropertyBought {
        player: u8,
        square: u8,
        price: u32,
    },
    PropertyAuctioned {
        square: u8,
        winner: u8,
        price: u32,
    },
    PropertyDeclined {
        player: u8,
        square: u8,
    },
    RentPaid {
        payer: u8,
        payee: u8,
        amount: u32,
        square: u8,
    },
    TaxPaid {
        player: u8,
        amount: u32,
        tax_kind: TaxKind,
    },
    CardDrawn {
        player: u8,
        is_chance: bool,
        effect: CardEffect,
    },
    HouseBuilt {
        player: u8,
        square: u8,
        houses: u8,
    },
    PropertyMortgaged {
        player: u8,
        square: u8,
        amount: u32,
    },
    PropertyUnmortgaged {
        player: u8,
        square: u8,
        cost: u32,
    },
    TradeOffered {
        proposer: u8,
        responder: u8,
    },
    TradeAccepted {
        proposer: u8,
        responder: u8,
    },
    TradeDeclined {
        proposer: u8,
        responder: u8,
    },
    PlayerJailed {
        player: u8,
        reason: JailReason,
    },
    PlayerReleasedFromJail {
        player: u8,
        method: ReleaseMethod,
    },
    PlayerBankrupt {
        player: u8,
        creditor: Option<u8>,
    },
    GameOver {
        winner: u8,
    },
    AuctionStarted {
        square: u8,
    },
    AuctionBid {
        player: u8,
        amount: u32,
    },
    AuctionWon {
        player: u8,
        square: u8,
        amount: u32,
    },
}

// ── Board Data Functions ───────────────────────────────────────

/// Returns the square kind for each of the 40 board positions.
pub const fn square_kind(index: u8) -> SquareKind {
    match index {
        0 => SquareKind::Go,
        1 => SquareKind::Property(PropertyGroup::Brown), // Mediterranean Ave
        2 => SquareKind::CommunityChest,
        3 => SquareKind::Property(PropertyGroup::Brown), // Baltic Ave
        4 => SquareKind::Tax(TaxKind::Income),
        5 => SquareKind::Railroad, // Reading Railroad
        6 => SquareKind::Property(PropertyGroup::LightBlue), // Oriental Ave
        7 => SquareKind::Chance,
        8 => SquareKind::Property(PropertyGroup::LightBlue), // Vermont Ave
        9 => SquareKind::Property(PropertyGroup::LightBlue), // Connecticut Ave
        10 => SquareKind::Jail,
        11 => SquareKind::Property(PropertyGroup::Pink), // St. Charles Place
        12 => SquareKind::Utility,                       // Electric Company
        13 => SquareKind::Property(PropertyGroup::Pink), // States Ave
        14 => SquareKind::Property(PropertyGroup::Pink), // Virginia Ave
        15 => SquareKind::Railroad,                      // Pennsylvania Railroad
        16 => SquareKind::Property(PropertyGroup::Orange), // St. James Place
        17 => SquareKind::CommunityChest,
        18 => SquareKind::Property(PropertyGroup::Orange), // Tennessee Ave
        19 => SquareKind::Property(PropertyGroup::Orange), // New York Ave
        20 => SquareKind::FreeParking,
        21 => SquareKind::Property(PropertyGroup::Red), // Kentucky Ave
        22 => SquareKind::Chance,
        23 => SquareKind::Property(PropertyGroup::Red), // Indiana Ave
        24 => SquareKind::Property(PropertyGroup::Red), // Illinois Ave
        25 => SquareKind::Railroad,                     // B&O Railroad
        26 => SquareKind::Property(PropertyGroup::Yellow), // Atlantic Ave
        27 => SquareKind::Property(PropertyGroup::Yellow), // Ventnor Ave
        28 => SquareKind::Utility,                      // Water Works
        29 => SquareKind::Property(PropertyGroup::Yellow), // Marvin Gardens
        30 => SquareKind::GoToJail,
        31 => SquareKind::Property(PropertyGroup::Green), // Pacific Ave
        32 => SquareKind::Property(PropertyGroup::Green), // North Carolina Ave
        33 => SquareKind::CommunityChest,
        34 => SquareKind::Property(PropertyGroup::Green), // Pennsylvania Ave
        35 => SquareKind::Railroad,                       // Short Line Railroad
        36 => SquareKind::Chance,
        37 => SquareKind::Property(PropertyGroup::DarkBlue), // Park Place
        38 => SquareKind::Tax(TaxKind::Luxury),
        39 => SquareKind::Property(PropertyGroup::DarkBlue), // Boardwalk
        _ => SquareKind::Go,
    }
}

/// Returns the property name for each of the 40 board positions.
pub fn square_name(index: u8) -> &'static str {
    match index {
        0 => "GO",
        1 => "Mediterranean Ave",
        2 => "Community Chest",
        3 => "Baltic Ave",
        4 => "Income Tax",
        5 => "Reading Railroad",
        6 => "Oriental Ave",
        7 => "Chance",
        8 => "Vermont Ave",
        9 => "Connecticut Ave",
        10 => "Jail",
        11 => "St. Charles Place",
        12 => "Electric Company",
        13 => "States Ave",
        14 => "Virginia Ave",
        15 => "Pennsylvania Railroad",
        16 => "St. James Place",
        17 => "Community Chest",
        18 => "Tennessee Ave",
        19 => "New York Ave",
        20 => "Free Parking",
        21 => "Kentucky Ave",
        22 => "Chance",
        23 => "Indiana Ave",
        24 => "Illinois Ave",
        25 => "B&O Railroad",
        26 => "Atlantic Ave",
        27 => "Ventnor Ave",
        28 => "Water Works",
        29 => "Marvin Gardens",
        30 => "Go To Jail",
        31 => "Pacific Ave",
        32 => "North Carolina Ave",
        33 => "Community Chest",
        34 => "Pennsylvania Ave",
        35 => "Short Line Railroad",
        36 => "Chance",
        37 => "Park Place",
        38 => "Luxury Tax",
        39 => "Boardwalk",
        _ => "Unknown",
    }
}

// ── Display Implementations ────────────────────────────────────

impl fmt::Display for PropertyGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Brown => "Brown",
            Self::LightBlue => "Light Blue",
            Self::Pink => "Pink",
            Self::Orange => "Orange",
            Self::Red => "Red",
            Self::Yellow => "Yellow",
            Self::Green => "Green",
            Self::DarkBlue => "Dark Blue",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for TurnPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::PreTurn => "Pre-Turn",
            Self::Rolling => "Rolling",
            Self::Resolving => "Resolving",
            Self::Acquisition => "Acquisition",
            Self::Auction => "Auction",
            Self::FinancialCrisis => "Financial Crisis",
            Self::Strategic => "Strategic",
            Self::EndTurn => "End Turn",
            Self::Bankrupt => "Bankrupt",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for TaxKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Income => "Income Tax",
            Self::Luxury => "Luxury Tax",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for JailReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::LandedOnGoToJail => "Landed on Go To Jail",
            Self::Speeding => "Speeding (3 doubles)",
            Self::CardEffect => "Card effect",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for ReleaseMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::PaidFine => "Paid fine",
            Self::UsedCard => "Used Get Out Of Jail Free card",
            Self::RolledDoubles => "Rolled doubles",
            Self::MaxTurnsExceeded => "Max jail turns exceeded",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for SquareKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Go => "GO",
            Self::Property(group) => return write!(f, "Property ({group})"),
            Self::Railroad => "Railroad",
            Self::Utility => "Utility",
            Self::Tax(kind) => return write!(f, "{kind}"),
            Self::Chance => "Chance",
            Self::CommunityChest => "Community Chest",
            Self::Jail => "Jail",
            Self::FreeParking => "Free Parking",
            Self::GoToJail => "Go To Jail",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for CardEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CollectMoney(amount) => write!(f, "Collect ${amount}"),
            Self::PayMoney(amount) => write!(f, "Pay ${amount}"),
            Self::PayPerHouse { house, hotel } => {
                write!(f, "Pay ${house}/house, ${hotel}/hotel")
            }
            Self::MoveTo(pos) => write!(f, "Move to {pos}"),
            Self::MoveBack(spaces) => write!(f, "Move back {spaces} spaces"),
            Self::MoveToNearest { is_railroad: true } => write!(f, "Move to nearest Railroad"),
            Self::MoveToNearest { is_railroad: false } => write!(f, "Move to nearest Utility"),
            Self::GoToJail => write!(f, "Go To Jail"),
            Self::GetOutOfJailFree => write!(f, "Get Out Of Jail Free"),
            Self::PayEachPlayer(amount) => write!(f, "Pay each player ${amount}"),
            Self::CollectFromEachPlayer(amount) => {
                write!(f, "Collect ${amount} from each player")
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_kind_covers_all_40_positions() {
        for i in 0..BOARD_SIZE {
            let kind = square_kind(i);
            assert!(
                format!("{kind}").len() > 0,
                "Square {i} should have a valid kind"
            );
        }
    }

    #[test]
    fn square_kind_specific_squares() {
        assert_eq!(square_kind(0), SquareKind::Go);
        assert_eq!(square_kind(10), SquareKind::Jail);
        assert_eq!(square_kind(20), SquareKind::FreeParking);
        assert_eq!(square_kind(30), SquareKind::GoToJail);
        assert_eq!(square_kind(4), SquareKind::Tax(TaxKind::Income));
        assert_eq!(square_kind(38), SquareKind::Tax(TaxKind::Luxury));
        assert_eq!(square_kind(5), SquareKind::Railroad);
        assert_eq!(square_kind(12), SquareKind::Utility);
        assert_eq!(square_kind(7), SquareKind::Chance);
        assert_eq!(square_kind(2), SquareKind::CommunityChest);
    }

    #[test]
    fn square_kind_property_groups() {
        // Brown: 1, 3
        assert!(matches!(
            square_kind(1),
            SquareKind::Property(PropertyGroup::Brown)
        ));
        assert!(matches!(
            square_kind(3),
            SquareKind::Property(PropertyGroup::Brown)
        ));
        // LightBlue: 6, 8, 9
        assert!(matches!(
            square_kind(6),
            SquareKind::Property(PropertyGroup::LightBlue)
        ));
        assert!(matches!(
            square_kind(8),
            SquareKind::Property(PropertyGroup::LightBlue)
        ));
        assert!(matches!(
            square_kind(9),
            SquareKind::Property(PropertyGroup::LightBlue)
        ));
        // DarkBlue: 37, 39
        assert!(matches!(
            square_kind(37),
            SquareKind::Property(PropertyGroup::DarkBlue)
        ));
        assert!(matches!(
            square_kind(39),
            SquareKind::Property(PropertyGroup::DarkBlue)
        ));
    }

    #[test]
    fn square_name_all_positions() {
        assert_eq!(square_name(0), "GO");
        assert_eq!(square_name(39), "Boardwalk");
        assert_eq!(square_name(10), "Jail");
        assert_eq!(square_name(255), "Unknown");
    }

    #[test]
    fn property_group_display() {
        assert_eq!(format!("{}", PropertyGroup::Brown), "Brown");
        assert_eq!(format!("{}", PropertyGroup::LightBlue), "Light Blue");
        assert_eq!(format!("{}", PropertyGroup::DarkBlue), "Dark Blue");
    }

    #[test]
    fn property_group_size() {
        assert_eq!(PropertyGroup::Brown.size(), 2);
        assert_eq!(PropertyGroup::DarkBlue.size(), 2);
        assert_eq!(PropertyGroup::LightBlue.size(), 3);
        assert_eq!(PropertyGroup::Green.size(), 3);
    }

    #[test]
    fn property_group_house_cost() {
        assert_eq!(PropertyGroup::Brown.house_cost(), 50);
        assert_eq!(PropertyGroup::Orange.house_cost(), 100);
        assert_eq!(PropertyGroup::Red.house_cost(), 150);
        assert_eq!(PropertyGroup::DarkBlue.house_cost(), 200);
    }

    #[test]
    fn property_group_all_has_8_members() {
        assert_eq!(PropertyGroup::all().len(), 8);
    }

    #[test]
    fn turn_phase_ordering() {
        let phases = [
            TurnPhase::PreTurn,
            TurnPhase::Rolling,
            TurnPhase::Resolving,
            TurnPhase::Acquisition,
            TurnPhase::Auction,
            TurnPhase::FinancialCrisis,
            TurnPhase::Strategic,
            TurnPhase::EndTurn,
            TurnPhase::Bankrupt,
        ];
        // Each phase should be distinct
        for (i, a) in phases.iter().enumerate() {
            for (j, b) in phases.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn turn_phase_display() {
        assert_eq!(format!("{}", TurnPhase::PreTurn), "Pre-Turn");
        assert_eq!(format!("{}", TurnPhase::Rolling), "Rolling");
        assert_eq!(format!("{}", TurnPhase::Bankrupt), "Bankrupt");
    }

    #[test]
    fn tax_kind_display() {
        assert_eq!(format!("{}", TaxKind::Income), "Income Tax");
        assert_eq!(format!("{}", TaxKind::Luxury), "Luxury Tax");
    }

    #[test]
    fn jail_reason_display() {
        assert_eq!(format!("{}", JailReason::Speeding), "Speeding (3 doubles)");
        assert_eq!(
            format!("{}", JailReason::LandedOnGoToJail),
            "Landed on Go To Jail"
        );
    }

    #[test]
    fn release_method_display() {
        assert_eq!(format!("{}", ReleaseMethod::PaidFine), "Paid fine");
        assert_eq!(
            format!("{}", ReleaseMethod::UsedCard),
            "Used Get Out Of Jail Free card"
        );
    }

    #[test]
    fn game_config_default_values() {
        let config = GameConfig::default();
        assert_eq!(config.starting_cash, STARTING_CASH);
        assert_eq!(config.salary, GO_SALARY);
        assert_eq!(config.jail_fine, JAIL_FINE);
        assert_eq!(config.max_jail_turns, MAX_JAIL_TURNS);
        assert_eq!(config.max_doubles, MAX_DOUBLES);
    }

    #[test]
    fn game_config_starting_cash() {
        let config = GameConfig::default();
        assert_eq!(config.starting_cash, 1500);
    }

    #[test]
    fn player_new_initial_state() {
        let player = Player::new(2, STARTING_CASH);
        assert_eq!(player.id, 2);
        assert_eq!(player.cash, STARTING_CASH);
        assert_eq!(player.position, GO_SQUARE);
        assert!(!player.in_jail);
        assert_eq!(player.jail_turns, 0);
        assert_eq!(player.get_out_of_jail_free, 0);
        assert_eq!(player.doubles_count, 0);
        assert!(!player.is_bankrupt);
    }

    #[test]
    fn player_pay_and_receive() {
        let mut player = Player::new(0, 500);
        assert!(player.pay(200));
        assert_eq!(player.cash, 300);
        assert!(!player.pay(400));
        assert_eq!(player.cash, 300); // unchanged after failed pay
        player.receive(100);
        assert_eq!(player.cash, 400);
    }

    #[test]
    fn trade_offer_creation() {
        let offer = TradeOffer::new(0, 2);
        assert_eq!(offer.proposer, 0);
        assert_eq!(offer.responder, 2);
        assert!(offer.proposer_properties.is_empty());
        assert!(offer.responder_properties.is_empty());
        assert_eq!(offer.proposer_cash, 0);
        assert_eq!(offer.responder_cash, 0);
    }

    #[test]
    fn trade_offer_net_value() {
        let mut offer = TradeOffer::new(0, 1);
        offer.proposer_cash = 200;
        offer.responder_cash = 300;
        // responder gives more cash, net value should be positive for proposer
        assert!(offer.net_value() > 0);
    }

    #[test]
    fn owned_component() {
        let entity = Entity::from_raw(42);
        let owned = Owned::new(entity);
        assert_eq!(owned.owner, entity);
        assert!(!owned.is_mortgaged);
        assert_eq!(owned.houses, 0);
        assert!(!owned.has_hotel());
        assert_eq!(owned.house_count(), 0);
    }

    #[test]
    fn owned_hotel_detection() {
        let entity = Entity::from_raw(1);
        let mut owned = Owned::new(entity);
        owned.houses = 5;
        assert!(owned.has_hotel());
        assert_eq!(owned.house_count(), 0);
        owned.houses = 4;
        assert!(!owned.has_hotel());
        assert_eq!(owned.house_count(), 4);
    }

    #[test]
    fn turn_state_advance_player() {
        let mut ts = TurnState::new(0);
        assert_eq!(ts.current_player, 0);
        assert_eq!(ts.turn_number, 1);
        ts.advance_player(4);
        assert_eq!(ts.current_player, 1);
        assert_eq!(ts.turn_number, 1);
        ts.advance_player(4);
        ts.advance_player(4);
        ts.advance_player(4);
        assert_eq!(ts.current_player, 0);
        assert_eq!(ts.turn_number, 2);
    }

    #[test]
    fn statistics_recording() {
        let mut stats = Statistics::default();
        stats.record_purchase(0);
        stats.record_purchase(2);
        assert_eq!(stats.properties_bought, [1, 0, 1, 0]);
        stats.record_rent(1, 50);
        stats.record_rent(1, 25);
        assert_eq!(stats.rent_paid, [0, 75, 0, 0]);
        stats.record_house(3);
        assert_eq!(stats.houses_built, [0, 0, 0, 1]);
        stats.record_trade(0);
        assert_eq!(stats.trades_completed, [1, 0, 0, 0]);
    }

    #[test]
    fn card_deck_draw() {
        let deck = CardDeck {
            cards: vec![CardEffect::CollectMoney(100), CardEffect::PayMoney(50)],
            draw_index: 0,
            is_chance: true,
        };
        let mut d = deck;
        let first = d.draw();
        assert!(matches!(first, CardEffect::CollectMoney(100)));
        let second = d.draw();
        assert!(matches!(second, CardEffect::PayMoney(50)));
    }

    #[test]
    fn card_effect_display() {
        assert_eq!(format!("{}", CardEffect::CollectMoney(200)), "Collect $200");
        assert_eq!(format!("{}", CardEffect::GoToJail), "Go To Jail");
        assert_eq!(
            format!(
                "{}",
                CardEffect::PayPerHouse {
                    house: 25,
                    hotel: 100
                }
            ),
            "Pay $25/house, $100/hotel"
        );
    }

    #[test]
    fn constants_are_sane() {
        assert_eq!(BOARD_SIZE, 40);
        assert_eq!(STARTING_CASH, 1500);
        assert_eq!(GO_SALARY, 200);
        assert_eq!(JAIL_FINE, 50);
        assert_eq!(JAIL_SQUARE, 10);
        assert_eq!(GO_TO_JAIL_SQUARE, 30);
        assert_eq!(FREE_PARKING_SQUARE, 20);
        assert_eq!(GO_SQUARE, 0);
        assert_eq!(INCOME_TAX_SQUARE, 4);
        assert_eq!(LUXURY_TAX_SQUARE, 38);
        assert_eq!(MAX_JAIL_TURNS, 3);
        assert_eq!(MAX_DOUBLES, 3);
        assert_eq!(MAX_HOUSES, 4);
        assert_eq!(AUCTION_MIN_BID, 10);
        assert!((MORTGAGE_INTEREST_RATE - 0.10).abs() < f32::EPSILON);
    }
}
