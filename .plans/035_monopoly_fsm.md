# Plan 035: Monopoly AI FSM — Turn-Based Event-Driven FSM with 4 Player Archetypes

**Branch:** `develop/feature/035_monopoly_fsm`
**Depends on:** Plan 033 (Bomberman Arena patterns), Plan 032 (HL Infrastructure), Plan 030 (Bandit)
**Status:** Planning

---

## Goal

Build a 4-player Monopoly game engine using `bevy_ecs` standalone with turn-based, event-driven FSM AI. Four AI players compete at progressively higher HL technology levels — same pattern as Bomberman (Plan 033), but adapted for a fundamentally different game genre: **turn-based financial strategy** instead of real-time spatial tactics.

The arena proves the FSM architecture handles both game genres with the same ECS + HL stack, and that the HL thesis extends beyond spatial domains: **adaptive financial strategy > static rules > random baselines**.

### Why Monopoly After Bomberman?

| Dimension | Bomberman (033) | Monopoly (035) |
|-----------|-----------------|----------------|
| Time model | Tick-based (200 ticks/round) | Turn-based (sequential phases) |
| Space model | 13×13 grid, continuous movement | 40-square board, discrete hops |
| Threat model | Spatial (blast zones, walls) | Financial (rent debt, cash flow) |
| Decision frequency | Every tick (6 actions) | Every turn phase (~8 decision points) |
| AI challenge | Real-time evasion + spatial planning | Financial planning + negotiation + risk management |
| FSM type | Per-tick state priority (Evade > Collect > Attack > Explore) | Per-turn phase sequence (PreTurn → Roll → Resolve → Strategic → End) |

This proves: **the same ECS + HL infrastructure scales across fundamentally different game genres**.

---

## Overview

4 AI players compete in a classic Monopoly game. Each player represents a rung on the HL technology ladder:

```
P1: Modelless (random)              — baseline (random legal actions)
P2: Model-based (greedy heuristic)   — buy everything, build aggressively
P3: Model + Validator (safety)       — keep cash reserves, avoid over-leverage
P4: Full HL (adaptive strategy)      — negotiation, adaptive building, bandit learning
```

### Architecture: bevy_ecs (Standalone) + ratatui TUI

Same stack as Bomberman — `bevy_ecs` standalone, ratatui TUI with emoji rendering. The difference is the game loop: **turn-based phase sequence** instead of tick-based simulation.

```
Bomberman (033)                     →  Monopoly (035)
────────────────────────────────────────────────────────
tick-based loop (200 ticks)         →  turn-based loop (sequential phases)
6 actions per tick                  →  1 action per phase decision point
spatial avoidance (blast zones)     →  financial avoidance (bankruptcy)
wall/grid collisions                →  property ownership / rent
power-up collection                 →  property acquisition / house building
bomb placement                      →  house/hotel building
opponent tracking (position)        →  opponent tracking (portfolio)
real-time evasion FSM               →  turn-phase FSM
```

---

## Monopoly Game Rules (Classic)

### Board Layout (40 Squares)

```text
┌─────────────────────────────────────────────────────────────┐
│  20 (Free Parking)  19  18  17  16  15  14  13  12  11  10 │
│                    [R]     [C]     [R]             [Jail]   │
│  21               18=NY  16=StCh  14=Virg  13=Stat  11=Con │
│                                                               │
│  22 [R]          Kentucky(21)       States(17)     NY(19)    │
│  23 Indiana                                             9 Con│
│  24 Illinois    ORANGE   ~~~~~~~~~~  RED          8 Vermont │
│  25 [R]          (props)              (props)      7 Chance │
│  26 Kentucky                                          6 [R] │
│  27 Chance                                             5 Read│
│  28 Park Place   GREEN    ~~~~~~~~~~  YELLOW        4 IncTax│
│  29 [R]          (props)              (props)       3 Baltic│
│  30 GoToJail                                            2 Comm│
│  31 Pacific                                             1 Med│
│  32 NC          DARK BLUE  ~~~~~~~~~~  LIGHT BLUE   0 GO   │
│  33 CommChest                                           +$200│
│  34 Park Place                                          │
│  35 LuxuryTax                                           │
│  36 Boardwalk                                           │
└─────────────────────────────────────────────────────────────┘
```

### Property Groups (Color Sets)

| Group | Properties | Base Rent | House Cost | Hotel Cost |
|-------|-----------|-----------|------------|------------|
| Brown | Mediterranean(1), Baltic(3) | $2/$4 | $50 | $50+4houses |
| Light Blue | Oriental(6), Vermont(8), Connecticut(9) | $6/$12 | $50 | $50+4houses |
| Pink | St.Charles(11), States(13), Virginia(14) | $10/$20 | $100 | $100+4houses |
| Orange | St.James(16), Tennessee(18), NewYork(19) | $14/$28 | $100 | $100+4houses |
| Red | Kentucky(21), Indiana(22), Illinois(23) | $18/$36 | $150 | $150+4houses |
| Yellow | Atlantic(26), Ventnor(27), Marvin(28) | $22/$44 | $150 | $150+4houses |
| Green | Pacific(31), NC(32), Pennsylvania(34) | $26/$52 | $200 | $200+4houses |
| Dark Blue | ParkPlace(37), Boardwalk(39) | $35/$70 | $200 | $200+4houses |

### Special Squares

| Square | Name | Effect |
|--------|------|--------|
| 0 | GO | Collect $200 salary |
| 5 | Income Tax | Pay $200 or 10% of net worth |
| 10 | Jail / Just Visiting | In jail or passing through |
| 12 | Electric Company | Utility — rent = 4×/10× dice roll |
| 20 | Free Parking | No effect (classic rules) |
| 28 | Water Works | Utility — rent = 4×/10× dice roll |
| 30 | Go To Jail | Move directly to jail |
| 38 | Luxury Tax | Pay $100 |

### Railroads (4)

Reading(5), Pennsylvania(15), B&O(25), ShortLine(35)
- Rent: $25 (1), $50 (2), $100 (3), $200 (4) — doubles with monopoly

### Dice & Movement

- Roll 2d6, move clockwise
- **Doubles:** Roll again (up to 3 times; 3rd doubles → Go To Jail)
- **Passing GO:** Collect $200

### Houses & Hotels

- Must own **complete color set** before building
- Build **evenly** — no house disparity > 1 within a color group
- Max 4 houses, then upgrade to hotel (returns 4 houses to bank)
- Housing shortage: if bank has <4 houses, auction them

### Mortgages

- Mortgage value = printed price / 2
- Unmortgage cost = mortgage value + 10% interest
- Mortgaged properties produce no rent
- Cannot build on a color group if any property is mortgaged

### Trading

- Players may trade properties, cash, and Get Out Of Jail Free cards
- Cannot trade houses/hotels — must sell back to bank first
- No self-trade (obviously)

### Auctions

- When a player lands on unowned property and declines to buy, it goes to auction
- All players (including decliner) may bid
- Minimum bid = $10, no upper limit
- Bidding continues until no one raises

### Bankruptcy

- When a player cannot pay a debt after liquidating all assets, they are bankrupt
- All properties transfer to the creditor (mortgaged properties stay mortgaged)
- Last player standing wins

---

## AI FSM Architecture (8 States)

Unlike Bomberman's priority-based per-tick FSM, Monopoly uses a **sequential phase-based FSM** — the AI progresses through states in order each turn.

### State Transition Diagram

```text
┌──────────────────────────────────────────────────────┐
│                                                       │
│  OTHER PLAYER'S TURN                                  │
│  ┌─────────┐   auction starts    ┌──────────┐        │
│  │  IDLE    │──────────────────→ │ AUCTION   │        │
│  │ (OffTurn)│                    │ (Bidding) │        │
│  └────┬─────┘                    └─────┬────┘        │
│       │ my turn starts                  │ auction ends │
│       ↓                                  ↓             │
│  ┌───────────┐  not in jail  ┌──────────┐            │
│  │ PRE-TURN  │─────────────→ │ ROLLING   │            │
│  │ (Jail Mgmt)│              │ (Dice)    │            │
│  └─────┬─────┘              └─────┬────┘            │
│        │ in jail, pay             │ dice result       │
│        │ or roll doubles          ↓                    │
│        └──────────────→  ┌──────────────┐            │
│                           │ RESOLVE      │            │
│                           │ (Landed Tile) │            │
│                           └──────┬───────┘            │
│                                  │                    │
│                    ┌─────────────┼─────────────┐      │
│                    ↓             ↓              ↓      │
│             ┌───────────┐ ┌──────────┐ ┌──────────┐  │
│             │ACQUISITION│ │  RENT/   │ │ FINANCIAL│  │
│             │(Buy/Pass) │ │  TAX/    │ │  CRISIS  │  │
│             └─────┬─────┘ │  CARD    │ │(Liquidate)│  │
│                   │       └────┬─────┘ └────┬─────┘  │
│                   │            │             │         │
│                   └────────────┴──────┬──────┘         │
│                                       ↓                │
│                              ┌──────────────┐         │
│                              │  STRATEGIC    │         │
│                              │ (Trade/Build) │         │
│                              └──────┬───────┘         │
│                                     │                  │
│                                     ↓                  │
│                              ┌──────────────┐         │
│                              │  END TURN     │         │
│                              └──────┬───────┘         │
│                                     │                  │
│                                     ↓                  │
│                              ┌──────────────┐         │
│                              │  BANKRUPT     │──→ END  │
│                              │ (Terminal)    │         │
│                              └──────────────┘         │
└──────────────────────────────────────────────────────┘
```

### State 1: Idle / Off-Turn

The AI waits for other players to finish their turns.

| Aspect | Detail |
|--------|--------|
| **Trigger** | It is another player's turn |
| **Actions** | Listen for trade offers, evaluate incoming auctions |
| **Trade Evaluation** | Does the trade complete a color set for AI? Does it break opponent's monopoly? |
| **Transitions** | Auction begins → **Auction**; My turn starts → **PreTurn** |

### State 2: PreTurn / Jail Management

Before rolling, handle current status and perform maintenance.

| Aspect | Detail |
|--------|--------|
| **Trigger** | AI's turn officially begins |
| **If in Jail (Early Game)** | Pay $50 or use Get Out Of Jail Free card — need to be on the board buying |
| **If in Jail (Late Game)** | Roll for doubles — board is dangerous, jail is safe |
| **If Not in Jail** | Check cash reserves; unmortgage if cash abundant; build houses on complete sets |
| **Transitions** | Ready to roll → **Rolling** |

### State 3: Rolling & Movement

Handle dice rolling and movement mechanics.

| Aspect | Detail |
|--------|--------|
| **Trigger** | Pre-turn maintenance complete |
| **Action** | Generate 2d6, check doubles, move token |
| **Speeding** | 3 doubles in a row → instant Jail, skip to **EndTurn** |
| **Transitions** | Sent to Jail → **EndTurn**; Landed on square → **ResolveSquare** |

### State 4: Resolve Square (Core Logic)

React to the tile landed on — the most complex state.

| Aspect | Detail |
|--------|--------|
| **Trigger** | Token lands on a new square |
| **Unowned Property** | Calculate cash buffer → **Acquisition** |
| **Opponent Property** | Calculate rent → pay or **FinancialCrisis** |
| **Own Property / Free Parking / Just Visiting** | No action |
| **Tax Space** | Deduct amount or **FinancialCrisis** |
| **Chance / Community Chest** | Draw card, execute effect (may re-trigger **ResolveSquare**) |
| **Go To Jail** | Move to Jail → **EndTurn** |
| **Transitions** | Depends on square type; ultimately → **Strategic** or **EndTurn** |

### State 5: Acquisition / Auction

Purchase decisions and bidding.

| Aspect | Detail |
|--------|--------|
| **Trigger** | Landed on unowned property, or auction triggered |
| **Direct Purchase** | Buy if cash - price ≥ cash_buffer (e.g., $200) |
| **Auction Bidding** | Calculate max bid based on strategic value; bid incrementally |
| **Strategic Value** | Completes color set (+50%), extends railroad count (+30%), standalone (-10%) |
| **Transitions** | Return to **ResolveSquare** flow or **EndTurn** |

### State 6: Financial Crisis / Liquidation

Survival mechanism when debt exceeds cash.

| Aspect | Detail |
|--------|--------|
| **Trigger** | Debt incurred exceeding current cash |
| **Step 1** | Sell houses/hotels back to bank (half price) |
| **Step 2** | Mortgage properties (prioritize singles without color matches) |
| **Step 3** | Attempt to trade assets for cash |
| **Transitions** | Debt covered → **EndTurn**; Cannot pay → **Bankrupt** |

### State 7: Strategic / Negotiation

Proactive gameplay before ending turn.

| Aspect | Detail |
|--------|--------|
| **Trigger** | Mandatory actions complete, turn not yet ended |
| **Trading** | Scan for monopoly-completing trades; offer cash/useless properties |
| **Building** | Build houses evenly on complete color sets if cash flow healthy |
| **Transitions** | → **EndTurn** |

### State 8: Bankrupt / Game Over

Terminal state.

| Aspect | Detail |
|--------|--------|
| **Trigger** | Debt > total liquidatable net worth |
| **Action** | Transfer all assets to creditor; disable AI token |
| **Transitions** | None — AI game loop terminates |

---

## AI Turn-Based Event Loop (Pseudocode)

```text
FUNCTION ExecuteTurn(player, game_state):
    // ── State 2: PreTurn ──
    IF player.in_jail:
        IF should_pay_out_of_jail(game_state):
            player.pay_jail_fine()
        ELSE:
            dice = roll_dice()
            IF dice.is_doubles:
                player.release_from_jail()
                GOTO ROLLING_WITH_RESULT(dice)
            ELSE:
                GOTO END_TURN

    // ── State 3: Rolling ──
    doubles_count = 0
    LOOP:
        dice = roll_dice()
        IF dice.is_doubles:
            doubles_count += 1
            IF doubles_count == 3:
                player.send_to_jail()  // Speeding!
                GOTO END_TURN
        ELSE:
            doubles_count = 0

        player.move_forward(dice.total)
        IF player.passed_go:
            player.collect(200)

        // ── State 4: Resolve Square ──
        square = board[player.position]
        MATCH square:
            CASE UnownedProperty:
                // ── State 5: Acquisition ──
                IF should_buy(player, square, game_state):
                    player.buy(square)
                ELSE:
                    start_auction(square)

            CASE OwnedBy(opponent):
                rent = calculate_rent(square, opponent, dice)
                IF player.cash < rent:
                    // ── State 6: Financial Crisis ──
                    IF liquidate_to_pay(player, rent):
                        player.pay(opponent, rent)
                    ELSE:
                        // ── State 8: Bankrupt ──
                        player.bankrupt_to(opponent)
                        RETURN GameOver

            CASE Tax:
                tax = calculate_tax(square, player)
                IF player.cash < tax:
                    IF NOT liquidate_to_pay(player, tax):
                        player.bankrupt_to(Bank)
                        RETURN GameOver
                ELSE:
                    player.pay_bank(tax)

            CASE CardDeck:
                card = draw_card(square)
                execute_card(card)  // May re-trigger movement

            CASE GoToJail:
                player.send_to_jail()
                GOTO END_TURN

            CASE OwnProperty | FreeParking | JustVisiting:
                pass  // No action

        IF dice.is_doubles AND player.active AND NOT player.in_jail:
            CONTINUE LOOP  // Roll again
        ELSE:
            BREAK LOOP

    // ── State 7: Strategic ──
    IF player.active:
        evaluate_trades(player, game_state)
        build_houses(player, game_state)

    // ── End Turn ──
    next_player()
```

---

## Player Types (4 HL Tech Levels)

### P1 🎲 RandomPlayer — Baseline

- **Tech:** None. Random legal decisions.
- **Buying:** 50% chance to buy if affordable.
- **Auction:** Random bid 0–50% of printed price.
- **Building:** Random houses on random valid properties.
- **Trading:** Declines all trades (simplest behavior).
- **Jail:** Always pays $50 if affordable.
- **No learning, no memory, no model.** Pure baseline.

### P2 💰 GreedyPlayer — Heuristic

- **Tech:** Heuristic scoring of all decisions.
- **Buying:** Buy everything affordable (cash - price ≥ $100 buffer).
- **Auction:** Bid up to 80% of strategic value.
- **Building:** Build houses on complete color sets, prioritize highest rent.
- **Trading:** Accept trades that increase property count or cash.
- **Jail:** Always pays $50 early (rounds 1–15), rolls late.
- **No opponent modeling, no risk assessment.**

### P3 🛡️ ValidatorPlayer — Heuristic + Safety Rules

- **Tech:** Same heuristic as P2, plus hard safety validation.
- **Buying:** Buy only if cash_buffer ≥ $200 after purchase (avoid over-leverage).
- **Auction:** Bid up to strategic value - safety margin.
- **Building:** Build only if cash remains ≥ $300 (rent reserve).
- **Trading:** Validate trades don't create opponent monopolies.
- **Jail:** Strategic — stay in jail late game when board is dangerous.
- **Safety rules:**
  - Never drop below minimum cash reserve
  - Never trade property that completes opponent's color set
  - Always keep enough cash to survive highest possible rent
- **Limitation:** Over-conservative, misses aggressive opportunities.

### P4 🧠 HLPlayer — Full HL (Adaptive Strategy + Bandit)

- **Tech:** P3 base + opponent portfolio tracking + adaptive strategy + bandit Q-values.
- **Tracks:** Opponent properties, cash estimates, color set completion likelihood.
- **Persists across games:** Q-values for strategies, compressed arms.

#### Strategy Adaptation

| Game Phase | Strategy | Trigger |
|-----------|----------|---------|
| **Early (rounds 1–10)** | Expansion — buy aggressively, complete sets | Most properties unowned |
| **Mid (rounds 11–25)** | Development — build houses, block trades | Color sets forming |
| **Late (rounds 26+)** | Survival — stay liquid, avoid rent traps | High-rent properties everywhere |

#### Opponent Modeling

- Track which properties each opponent owns
- Estimate opponent cash from observed transactions
- Calculate opponent's "threat level" (highest possible rent × monopoly multiplier)
- Predict which trades opponents would accept

#### Financial Optimization

- **Cash flow analysis:** Expected rent income vs rent exposure per lap
- **Monopoly completion value:** Quantify how much a missing property is worth
- **Mortgage timing:** Mortgage low-value singles to fund high-value development
- **House lock strategy:** Buy houses to create housing shortage for opponents

#### Bandit Layer

- **Arms:** Strategy profiles (expansion / development / survival / aggressive / conservative)
- **Reward:** +1.0 survive a round, -1.0 go bankrupt, +0.5 complete monopoly, +0.3 bankrupt opponent
- **Absorb-Compress:** Every 10 games, compress strategies with Q < 0.1

---

## ECS Architecture (bevy_ecs standalone)

### Components

```rust
// ── Player ─────────────────────────────────────────────────
#[derive(Component)]
struct Player {
    id: u8,
    cash: u32,
    position: u8,          // 0–39 board index
    in_jail: bool,
    jail_turns: u8,         // How many turns in jail (max 3)
    get_out_of_jail_free: u8, // Count of GOOJF cards held
    doubles_count: u8,      // Consecutive doubles this turn
    is_bankrupt: bool,
}

// ── Property Ownership ─────────────────────────────────────
#[derive(Component)]
struct Property {
    square: u8,             // Board position (0–39)
    group: PropertyGroup,   // Color group or special type
    name: &'static str,
    price: u32,             // Purchase price
    base_rent: u32,         // Rent with no houses
    house_cost: u32,        // Cost per house
    house_rent: [u32; 5],   // Rent with 1–4 houses + hotel
    mortgage_value: u32,
}

#[derive(Component)]
struct Owned {
    owner: Entity,
    is_mortgaged: bool,
    houses: u8,             // 0–4 houses, 5 = hotel
}

// ── Board Square ───────────────────────────────────────────
#[derive(Component)]
struct BoardSquare {
    index: u8,              // 0–39
    kind: SquareKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SquareKind {
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
enum PropertyGroup {
    Brown,
    LightBlue,
    Pink,
    Orange,
    Red,
    Yellow,
    Green,
    DarkBlue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TaxKind {
    Income,   // $200 or 10%
    Luxury,   // $100
}

// ── Cards ───────────────────────────────────────────────────
#[derive(Clone, Debug)]
enum CardEffect {
    CollectMoney(u32),
    PayMoney(u32),
    PayPerHouse { house: u32, hotel: u32 },
    MoveTo(u8),
    MoveBack(u8),
    MoveToNearest(SquareKind),
    GoToJail,
    GetOutOfJailFree,
    PayEachPlayer(u32),
    CollectFromEachPlayer(u32),
}

#[derive(Component)]
struct CardDeck {
    cards: Vec<CardEffect>,
    draw_index: usize,
    is_chance: bool,
}
```

### Resources

```rust
#[derive(Resource)]
struct Board {
    squares: [Entity; 40],  // Entity refs for each square
}

#[derive(Resource)]
struct TurnState {
    current_player: u8,
    phase: TurnPhase,
    turn_number: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurnPhase {
    PreTurn,
    Rolling,
    Resolving,
    Acquisition,
    Auction { highest_bidder: Option<u8>, highest_bid: u32 },
    FinancialCrisis { debt: u32, creditor: Option<Entity> },
    Strategic,
    EndTurn,
}

#[derive(Resource)]
struct GameConfig {
    starting_cash: u32,     // Default: $1500
    salary: u32,            // Default: $200
    jail_fine: u32,         // Default: $50
    max_jail_turns: u8,     // Default: 3
    double_go_to_jail: u8,  // Default: 3 doubles
}

#[derive(Resource)]
struct PlayerEntities {
    entities: [Entity; 4],
}

#[derive(Resource, Default)]
struct Statistics {
    rounds_completed: u32,
    properties_bought: [u32; 4],
    rent_paid: [u32; 4],
    houses_built: [u32; 4],
    trades_completed: [u32; 4],
}
```

### Events

```rust
#[derive(Event, Clone, Debug)]
enum GameEvent {
    TurnStarted { player: u8 },
    DiceRolled { player: u8, die1: u8, die2: u8, doubles: bool },
    PlayerMoved { player: u8, from: u8, to: u8, passed_go: bool },
    SalaryCollected { player: u8, amount: u32 },
    PropertyBought { player: u8, square: u8, price: u32 },
    PropertyAuctioned { square: u8, winner: u8, price: u32 },
    RentPaid { payer: u8, payee: u8, amount: u32, square: u8 },
    TaxPaid { player: u8, amount: u32, tax_kind: TaxKind },
    CardDrawn { player: u8, is_chance: bool, effect: CardEffect },
    HouseBuilt { player: u8, square: u8, houses: u8 },
    HotelBuilt { player: u8, square: u8 },
    PropertyMortgaged { player: u8, square: u8, amount: u32 },
    PropertyUnmortgaged { player: u8, square: u8, cost: u32 },
    TradeOffered { proposer: u8, responder: u8, offer: TradeOffer },
    TradeAccepted { proposer: u8, responder: u8 },
    TradeDeclined { proposer: u8, responder: u8 },
    PlayerJailed { player: u8, reason: JailReason },
    PlayerReleasedFromJail { player: u8, method: ReleaseMethod },
    PlayerBankrupt { player: u8, creditor: Option<u8> },
    GameOver { winner: u8 },
}

#[derive(Clone, Debug)]
struct TradeOffer {
    proposer_gives_properties: Vec<u8>,
    proposer_gives_cash: u32,
    responder_gives_properties: Vec<u8>,
    responder_gives_cash: u32,
}

#[derive(Clone, Copy, Debug)]
enum JailReason {
    LandedOnGoToJail,
    Speeding,      // 3 doubles
    CardEffect,
}

#[derive(Clone, Copy, Debug)]
enum ReleaseMethod {
    PaidFine,
    UsedCard,
    RolledDoubles,
    MaxTurnsExceeded,
}
```

### Systems

```rust
fn init_game(seed: u64) -> World;
fn spawn_players(world: &mut World) -> [Entity; 4];
fn build_board(world: &mut World);
fn shuffle_decks(world: &mut World, seed: u64);

// Turn execution (called per active player)
fn execute_turn(world: &mut World, actions: &mut [Box<dyn MonopolyPlayer>; 4], rng: &mut Rng) -> TurnResult;

// Phase systems (called within execute_turn)
fn phase_pre_turn(world: &mut World, player: Entity) -> TurnPhase;
fn phase_rolling(world: &mut World, player: Entity) -> TurnPhase;
fn phase_resolve(world: &mut World, player: Entity, square: u8, dice: (u8, u8)) -> TurnPhase;
fn phase_acquisition(world: &mut World, player: Entity, square: u8) -> TurnPhase;
fn phase_auction(world: &mut World, square: u8, players: &mut [Box<dyn MonopolyPlayer>; 4]) -> TurnPhase;
fn phase_financial_crisis(world: &mut World, player: Entity, debt: u32, creditor: Option<Entity>) -> TurnPhase;
fn phase_strategic(world: &mut World, player: Entity) -> TurnPhase;

// Utility systems
fn calculate_rent(world: &World, square: u8, dice: (u8, u8)) -> u32;
fn calculate_net_worth(world: &World, player: Entity) -> u32;
fn owns_complete_set(world: &World, player: Entity, group: PropertyGroup) -> bool;
fn count_houses_in_set(world: &World, player: Entity, group: PropertyGroup) -> u8;
fn can_build_house(world: &World, player: Entity, square: u8) -> bool;
fn liquidate_assets(world: &mut World, player: Entity, target: u32) -> u32;
fn transfer_assets(world: &mut World, from: Entity, to: Entity);
```

### App Setup

```rust
fn monopoly_app(seed: u64) -> World {
    let mut world = World::new();
    world.insert_resource(GameConfig::default());
    world.insert_resource(TurnState::default());
    world.insert_resource(Statistics::default());
    world.init_resource::<Events<GameEvent>>();
    build_board(&mut world);
    shuffle_decks(&mut world, seed);
    spawn_players(&mut world);
    world
}
```

---

## TUI Rendering (ratatui + emoji)

### Board Emoji Map

| Cell | Emoji |
|------|-------|
| GO | 🏁 |
| Brown Property | 🟤 |
| Light Blue Property | 🔵 |
| Pink Property | 🩷 |
| Orange Property | 🟠 |
| Red Property | 🔴 |
| Yellow Property | 🟡 |
| Green Property | 🟢 |
| Dark Blue Property | 🔷 |
| Railroad | 🚂 |
| Utility | ⚡ |
| Tax | 📋 |
| Chance | ❓ |
| Community Chest | 📦 |
| Jail | 🔒 |
| Free Parking | 🅿️ |
| Go To Jail | 👮 |
| Player Token | 🎲/💰/🛡️/🧠 |

### TUI Layout

```text
┌─ Monopoly FSM Arena ─────────────────────────────────────────────────┐
│ Board (40 squares around perimeter)                                   │
│ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                                  │
│ │🅿│🟢│🟢│🟢│🚂│🔵│❓│🔵│🔵│🔒│ ← Top row (20-29)              │
│ ├──┤                          ├──┤                                    │
│ │🚂                          │❓│                                    │
│ ├──┤                          ├──┤                                    │
│ │🟡                          │🔵│                                    │
│ ├──┤   Player Stats Area      ├──┤                                    │
│ │🟡   P1 🎲 $1200 3 props    │🟤│                                    │
│ ├──┤   P2 💰 $950  5 props    ├──┤                                    │
│ │🚂   P3 🛡️ $800  4 props    │📋│                                    │
│ ├──┤   P4 🧠 $1100 6 props    ├──┤                                    │
│ │❓                          │🟤│                                    │
│ ├──┤                          ├──┤                                    │
│ │🟠                          │🏁│                                    │
│ ├──┤                          ├──┤                                    │
│ │🟠                          │🟠│                                    │
│ ├──┤                          ├──┤                                    │
│ │🚂                          │🚂│                                    │
│ ├──┤                          ├──┤                                    │
│ │🟣                          │📦│                                    │
│ ├──┤                          ├──┤                                    │
│ │📋                          │🟣│                                    │
│ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                                  │
│ ↑ Bottom row (10-0, right to left)                                   │
│                                                                       │
│ Event Log:                                                            │
│ > P2 rolled 3+4=7, landed on Kentucky 🔴 → Paid $18 rent to P4      │
│ > P4 builds 1 house on Park Place 🔷                                 │
└───────────────────────────────────────────────────────────────────────┘
```

### Controls (same as bomber_02_tui.rs)

| Key | Action |
|-----|--------|
| `Space` | Next event / next turn |
| `→` | Fast forward 10 turns |
| `F` | Fast forward to game end |
| `Q` | Quit |

---

## Tasks

- [ ] **Task 1: Core Types & Board Data** (`src/pruners/monopoly/mod.rs`)
  - Define `PropertyGroup`, `SquareKind`, `TaxKind`, `TurnPhase` enums
  - Define `CardEffect` enum with all classic card types
  - Define `GameEvent` enum with all game events
  - Define `TradeOffer`, `JailReason`, `ReleaseMethod` types
  - Define all 40 board squares as static data (name, price, rent, group)
  - Define Chance and Community Chest card decks (classic 16 each)
  - Unit tests for enum conversions and board data integrity

- [ ] **Task 2: ECS Components & Resources** (`src/pruners/monopoly/mod.rs`)
  - `Player` component with cash, position, jail state, GOOJF cards
  - `Property` component with square data (name, price, rent table, house cost)
  - `Owned` component (owner, mortgage, house count)
  - `BoardSquare` component for each square entity
  - `CardDeck` component for Chance/Community Chest
  - `Board`, `TurnState`, `GameConfig`, `PlayerEntities`, `Statistics` resources
  - Unit tests for component defaults

- [ ] **Task 3: Board Initialization** (`src/pruners/monopoly/board.rs`)
  - `build_board(world)` — create 40 square entities with correct Property data
  - `shuffle_decks(world, seed)` — shuffle Chance and Community Chest
  - Property data: all 22 streets, 4 railroads, 2 utilities, 6 special squares
  - Rent tables: base rent, monopoly rent (double), 1–4 houses, hotel
  - Verify all 40 squares have correct indices, prices, and group assignments

- [ ] **Task 4: Game Systems** (`src/pruners/monopoly/systems.rs`)
  - `init_game(seed)` — create world with all resources and board
  - `spawn_players(world)` — 4 players with $1500 starting cash
  - `phase_pre_turn` — jail management, pre-roll decisions
  - `phase_rolling` — 2d6 roll, doubles tracking, movement, passing GO
  - `phase_resolve` — square-type dispatch (property, tax, card, jail, etc.)
  - `phase_acquisition` — buy decision or trigger auction
  - `phase_auction` — bidding loop until winner
  - `phase_financial_crisis` — liquidate houses, mortgage, check bankruptcy
  - `phase_strategic` — building and trading decisions
  - Utility functions: `calculate_rent`, `calculate_net_worth`, `owns_complete_set`, `can_build_house`, `liquidate_assets`, `transfer_assets`
  - Card effect execution (move, collect, pay, jail, etc.)
  - Unit tests for each phase with fixed dice

- [ ] **Task 5: MonopolyPlayer Trait** (`src/pruners/monopoly/players.rs`)
  - `trait MonopolyPlayer` with decision methods:
    - `should_buy_property(&self, ctx: &DecisionContext) -> bool`
    - `auction_bid(&mut self, ctx: &AuctionContext) -> u32`
    - `jail_decision(&self, ctx: &DecisionContext) -> JailDecision`
    - `build_decision(&self, ctx: &DecisionContext) -> Vec<u8>` (squares to build on)
    - `trade_response(&mut self, offer: &TradeOffer, ctx: &DecisionContext) -> TradeResponse`
    - `propose_trade(&self, ctx: &DecisionContext) -> Option<TradeOffer>`
    - `mortgage_priority(&self, ctx: &DecisionContext) -> Vec<u8>` (order to mortgage)
    - `name(&self) -> &str`, `emoji(&self) -> &str`, `reset(&mut self)`
  - `DecisionContext` struct with read-only game state for AI decisions
  - `JailDecision` enum: `PayFine`, `UseCard`, `RollForDoubles`
  - `TradeResponse` enum: `Accept`, `Decline`, `CounterOffer(TradeOffer)`

- [ ] **Task 6: RandomPlayer (P1)** (`src/pruners/monopoly/players.rs`)
  - Buy: 50% chance if affordable
  - Auction: random bid 0–50% of price
  - Jail: always pay if affordable
  - Build: random valid house placement
  - Trade: always decline
  - Mortgage: random order
  - Unit tests verifying random decisions stay within legal bounds

- [ ] **Task 7: GreedyPlayer (P2)** (`src/pruners/monopoly/players.rs`)
  - Buy: everything affordable (cash - price ≥ $100)
  - Auction: bid up to 80% of strategic value
  - Jail: pay early (rounds 1–15), roll late
  - Build: highest rent properties first on complete sets
  - Trade: accept if increases property count or cash
  - Mortgage: least valuable first
  - Heuristic scoring function for properties
  - Unit tests for buying/building priorities

- [ ] **Task 8: ValidatorPlayer (P3)** (`src/pruners/monopoly/players.rs`)
  - Buy: cash_buffer ≥ $200 after purchase
  - Auction: bid up to strategic value minus safety margin
  - Jail: stay late game (board dangerous), pay early
  - Build: only if cash remains ≥ $300
  - Trade: validate no opponent monopoly creation
  - Safety rules: minimum cash reserve, rent exposure limits
  - Financial risk assessment function
  - Unit tests for safety constraints (never drops below reserve)

- [ ] **Task 9: HLPlayer (P4)** (`src/pruners/monopoly/players.rs`)
  - All P3 safety rules + opponent portfolio tracking
  - Game phase detection (early/mid/late) with strategy adaptation
  - Opponent modeling: track properties, estimate cash, calculate threat levels
  - Financial optimization: cash flow analysis, monopoly completion value
  - Bandit layer: strategy arms with Q-values, persist across games
  - Absorb-compress: every 10 games, compress low-Q strategies
  - Advanced tactics: house lock (create housing shortage), mortgage timing
  - Unit tests for opponent modeling and strategy selection

- [ ] **Task 10: Headless Arena Example** (`examples/monopoly_01_arena.rs`)
  - Run N games (default: 100) with 4 players
  - Per-game results: winner, turns played, final cash, properties owned
  - Cumulative standings: wins, avg cash, avg net worth, bankruptcies
  - Event scoping: turn-scoped events for AI, accumulated for stats
  - Output format matching bomber_01_arena.rs style
  - Configurable seed for reproducibility

- [ ] **Task 11: TUI Example** (`examples/monopoly_02_tui.rs`)
  - ratatui board rendering with emoji
  - Player stats panel (cash, properties, houses)
  - Event log with scrollable history
  - Turn-by-turn or fast-forward controls
  - Color-coded properties by group
  - House/hotel indicators on property squares

- [ ] **Task 12: HL Proof Example** (`examples/monopoly_03_hl_proof.rs`)
  - Run 1000 games comparing HL vs Validator vs Greedy vs Random
  - Metrics: win rate, survival rate, avg turns to win, avg net worth at end
  - Statistical significance: is HL > Validator by ≥5pp win rate?
  - Golden trace output for regression detection
  - Bandit learning visualization (Q-value convergence across games)

- [ ] **Task 13: Tests & Docs**
  - Unit tests: board data integrity, rent calculation, building rules, trade validation
  - Integration tests: full game from start to bankruptcy, doubles mechanics, auction flow
  - Edge cases: all properties owned, housing shortage, 3 players bankrupt, trade loops
  - Update `.docs/11_monopoly_fsm.md` with architecture and results
  - Update `.docs/01_overview.md` with monopoly module listing
  - Update `README.md` with monopoly arena section

---

## Cargo.toml Changes

```toml
[features]
monopoly = ["bevy_ecs", "bandit"]  # Monopoly FSM Arena (Plan 035)

[dependencies]
# No new dependencies — uses same bevy_ecs + fastrand + bandit as bomber

[[example]]
name = "monopoly_01_arena"
required-features = ["monopoly"]

[[example]]
name = "monopoly_02_tui"
required-features = ["monopoly"]

[[example]]
name = "monopoly_03_hl_proof"
required-features = ["monopoly"]
```

---

## Module Structure

```
src/pruners/monopoly/
├── mod.rs           # Types, enums, components, resources, events, constants
├── board.rs         # Board initialization, 40 squares data, deck shuffling
├── systems.rs       # Game systems: init, phases, utilities
└── players.rs       # MonopolyPlayer trait + 4 implementations

examples/
├── monopoly_01_arena.rs   # Headless 100-game tournament
├── monopoly_02_tui.rs     # Animated ratatui TUI replay
└── monopoly_03_hl_proof.rs # 1000-game HL proof experiment

tests/
└── bench_monopoly.rs      # Performance benchmarks
```

---

## File Locations

| File | Est. Lines | Purpose |
|------|-----------|---------|
| `src/pruners/monopoly/mod.rs` | ~400 | Enums, components, resources, events, board data constants |
| `src/pruners/monopoly/board.rs` | ~250 | Board init, deck shuffle, property definitions |
| `src/pruners/monopoly/systems.rs` | ~700 | Turn execution, phase systems, utility functions |
| `src/pruners/monopoly/players.rs` | ~1200 | Trait + 4 AI implementations |
| `examples/monopoly_01_arena.rs` | ~250 | Headless tournament runner |
| `examples/monopoly_02_tui.rs` | ~550 | TUI with board rendering |
| `examples/monopoly_03_hl_proof.rs` | ~450 | 1000-game proof experiment |

---

## Expected Results

### Win Rate (100 Games)

```text
#1 🧠 HL          Wins=~30  Avg Net Worth=~$3500  Bankruptcies=~5
#2 🛡️ Validator   Wins=~25  Avg Net Worth=~$2800  Bankruptcies=~10
#3 💰 Greedy      Wins=~25  Avg Net Worth=~$2600  Bankruptcies=~20
#4 🎲 Random      Wins=~20  Avg Net Worth=~$1800  Bankruptcies=~30
```

### Key Predictions

1. **HL > Validator by ≥5pp** — adaptive strategy beats static rules in a game with as much variance as Monopoly
2. **Greedy dies more than Validator** — aggressive buying without safety margins leads to bankruptcy when rent hits
3. **Random occasionally wins** — Monopoly has high variance; lucky dice can overcome bad strategy
4. **Validator survives longest** — conservative cash management means fewest bankruptcies
5. **Score ≠ Wins** — Greedy may have higher average net worth (buys everything) but also goes bankrupt more

### Performance

| Metric | Target |
|--------|--------|
| Full game (avg 50 turns × 4 players) | < 100ms headless |
| AI decision per turn | < 1ms |
| TUI rendering per frame | < 16ms (60fps) |
| 1000-game proof | < 2 minutes |

---

## Design Lessons (Expected)

1. **Phase-based FSM ≠ Priority-based FSM** — Monopoly's sequential phases differ fundamentally from Bomberman's priority-ordered states, but both work with the same ECS + trait architecture
2. **Financial risk is a different threat model** — instead of spatial blast zones, the AI manages cash reserves and debt exposure
3. **Opponent modeling scales in complexity** — Bomberman tracks position + trajectory; Monopoly tracks entire portfolio + cash estimates + trade propensity
4. **Variance requires more games for proof** — Monopoly's dice-driven gameplay means 1000 games (not 100) for statistical significance
5. **Negotiation is the hardest AI challenge** — evaluating trade fairness requires understanding both sides' portfolio completion state

---

## Out of Scope

- Human player input (AI-only arena)
- Property auctions with housing shortage resolution
- Complex house trading rules (variant rules)
- Speed die (Mega Monopoly variant)
- Multi-game tournament bracket
- WASM validator (deferred to Plan 036, same pattern as Plan 034)
- LoRA model training for property valuation (deferred)
- Network multiplayer

---

## References

- Plan 033: Bomberman Arena (ECS + HL patterns, the template for this plan)
- Plan 034: Bomber WASM Validator (future WASM adaptation for Monopoly)
- Plan 032: HL Infrastructure (BanditPruner, HotSwapPruner, TrialLog)
- Plan 030: Multi-Armed Bandit (bandit Q-values, ε-greedy, absorb-compress)
- `.docs/10_bomber_arena.md` — current Bomberman documentation
- Classic Monopoly rules (Hasbro/Parker Brothers)