//! Monopoly Board Game Engine — ECS game systems (bevy_ecs 0.15, World-based).
//!
//! All systems operate on `&mut World` for turn-based game loop control.
//! No ECS schedule — functions are called in deterministic order by the game loop.

use bevy_ecs::prelude::*;

use super::board::{build_board, group_squares, shuffle_decks};
use super::players::{DecisionContext, MonopolyPlayer};
use super::{
    BOARD_SIZE, Board, BoardSquare, CardDeck, CardEffect, GameConfig, GameEvent, JAIL_SQUARE,
    JailDecision, JailReason, MAX_DOUBLES, MAX_HOUSES, MAX_JAIL_TURNS, Owned, Player,
    PlayerEntities, Property, PropertyGroup, ReleaseMethod, SquareKind, Statistics, TaxKind,
    TurnState,
};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a single player's turn.
#[derive(Debug)]
pub struct TurnResult {
    pub player: u8,
    pub events: Vec<GameEvent>,
    pub went_bankrupt: bool,
}

/// Result of a complete game.
#[derive(Debug)]
pub struct GameResult {
    pub winner: u8,
    pub total_turns: u32,
    pub events: Vec<GameEvent>,
}

// ---------------------------------------------------------------------------
// World initialisation
// ---------------------------------------------------------------------------

/// Create a fresh World with all resources, board, card decks, and 4 players.
pub fn init_world(seed: u64) -> World {
    let mut world = World::new();
    world.insert_resource(GameConfig::default());
    world.insert_resource(TurnState::new(0));
    world.insert_resource(Statistics::default());
    world.init_resource::<Events<GameEvent>>();
    build_board(&mut world);
    shuffle_decks(&mut world, seed);
    let entities = spawn_players(&mut world);
    world.insert_resource(PlayerEntities { entities });
    world
}

/// Spawn 4 player entities at GO (position 0) with starting cash.
pub fn spawn_players(world: &mut World) -> [Entity; 4] {
    let starting = world.resource::<GameConfig>().starting_cash;
    let entities: Vec<Entity> = (0..4)
        .map(|i| world.spawn(Player::new(i, starting)).id())
        .collect();
    let [a, b, c, d] = entities.try_into().expect("4 players");
    [a, b, c, d]
}

// ---------------------------------------------------------------------------
// Decision context builder
// ---------------------------------------------------------------------------

/// Build a [`DecisionContext`] snapshot from the current world state.
fn build_ctx(world: &World, player_id: u8, turn_number: u32) -> DecisionContext {
    let player_entities = world.resource::<PlayerEntities>().entities;
    let entity = player_entities[player_id as usize];
    let player = world.get::<Player>(entity).expect("player exists");
    let squares = world.resource::<Board>().squares;

    let mut owned_properties = Vec::new();
    let mut group_counts = [0u8; 8];
    let mut opponent_cash = [0u32; 4];
    let mut opponent_property_count = [0u8; 4];
    let mut square_owners = [None; 40];
    let mut square_houses_arr = [0u8; 40];
    let mut square_mortgaged = [false; 40];
    let mut square_prices = [0u32; 40];
    let mut square_base_rent = [0u32; 40];
    let mut square_house_cost = [0u32; 40];
    let mut square_mortgage_value = [0u32; 40];

    for sq_idx in 0..BOARD_SIZE as usize {
        let sq_entity = squares[sq_idx];
        if let Some(prop) = world.get::<Property>(sq_entity) {
            square_prices[sq_idx] = prop.price;
            square_base_rent[sq_idx] = prop.base_rent;
            square_house_cost[sq_idx] = prop.house_cost;
            square_mortgage_value[sq_idx] = prop.mortgage_value;
        }
        if let Some(owned) = world.get::<Owned>(sq_entity) {
            let owner_id = pid_from_entity_arr(&player_entities, owned.owner);
            square_owners[sq_idx] = Some(owner_id);
            square_houses_arr[sq_idx] = owned.houses;
            square_mortgaged[sq_idx] = owned.is_mortgaged;
            if owned.owner == entity {
                owned_properties.push(sq_idx as u8);
                // Only count actual street properties in group_counts
                // (railroads/utilities have Property with placeholder Brown group)
                let is_street = world
                    .get::<BoardSquare>(sq_entity)
                    .map(|bs| matches!(bs.kind, SquareKind::Property(_)))
                    .unwrap_or(false);
                if is_street
                    && let Some(prop) = world.get::<Property>(sq_entity)
                    && (prop.group as usize) < group_counts.len()
                {
                    group_counts[prop.group as usize] += 1;
                }
            } else {
                let idx = owner_id as usize;
                if idx < 4 {
                    opponent_property_count[idx] += 1;
                }
            }
        }
    }

    for i in 0..4u8 {
        if i != player_id {
            let e = player_entities[i as usize];
            opponent_cash[i as usize] = world.get::<Player>(e).map(|p| p.cash).unwrap_or(0);
        }
    }

    DecisionContext {
        player_id,
        cash: player.cash,
        position: player.position,
        owned_properties,
        group_counts,
        opponent_cash,
        opponent_property_count,
        square_owners,
        square_houses: square_houses_arr,
        square_mortgaged,
        square_prices,
        square_base_rent,
        square_house_cost,
        square_mortgage_value,
        turn_number,
        in_jail: player.in_jail,
        jail_turns: player.jail_turns,
        has_jail_card: player.get_out_of_jail_free > 0,
    }
}

/// Look up player id from entity using a pre-fetched entity array.
fn pid_from_entity_arr(entities: &[Entity; 4], entity: Entity) -> u8 {
    for (i, &e) in entities.iter().enumerate() {
        if e == entity {
            return i as u8;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Player helpers
// ---------------------------------------------------------------------------

fn is_player_active(world: &World, id: u8) -> bool {
    let pe = world.resource::<PlayerEntities>();
    let entity = pe.entities[id as usize];
    world
        .get::<Player>(entity)
        .map(|p| !p.is_bankrupt)
        .unwrap_or(false)
}

fn count_active_players(world: &World) -> u8 {
    let pe = world.resource::<PlayerEntities>();
    let mut count = 0u8;
    for i in 0..4u8 {
        let entity = pe.entities[i as usize];
        if world
            .get::<Player>(entity)
            .map(|p| !p.is_bankrupt)
            .unwrap_or(false)
        {
            count += 1;
        }
    }
    count
}

fn find_winner(world: &World) -> u8 {
    for i in 0..4u8 {
        if is_player_active(world, i) {
            return i;
        }
    }
    0
}

fn find_richest(world: &World) -> u8 {
    let pe = world.resource::<PlayerEntities>();
    let mut best = 0u8;
    let mut best_nw = 0u32;
    for i in 0..4u8 {
        if is_player_active(world, i) {
            let entity = pe.entities[i as usize];
            let nw = calculate_net_worth(world, entity);
            if nw > best_nw {
                best_nw = nw;
                best = i;
            }
        }
    }
    best
}

fn player_entity(world: &World, id: u8) -> Entity {
    world.resource::<PlayerEntities>().entities[id as usize]
}

fn player_id_from_entity(world: &World, entity: Entity) -> u8 {
    let pe = world.resource::<PlayerEntities>();
    for (i, &e) in pe.entities.iter().enumerate() {
        if e == entity {
            return i as u8;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Movement
// ---------------------------------------------------------------------------

fn roll_dice(rng: &mut fastrand::Rng) -> (u8, u8, bool) {
    let d1 = rng.u8(1..=6);
    let d2 = rng.u8(1..=6);
    (d1, d2, d1 == d2)
}

fn move_forward(world: &mut World, entity: Entity, steps: u8) -> (u8, bool) {
    let old = world.get::<Player>(entity).map(|p| p.position).unwrap_or(0);
    let new = ((old as u16 + steps as u16) % BOARD_SIZE as u16) as u8;
    let passed_go = new < old;
    if let Some(mut p) = world.get_mut::<Player>(entity) {
        p.position = new;
    }
    (new, passed_go)
}

fn move_to(world: &mut World, entity: Entity, target: u8) {
    if let Some(mut p) = world.get_mut::<Player>(entity) {
        p.position = target;
    }
}

fn collect_salary(world: &mut World, entity: Entity, events: &mut Vec<GameEvent>) {
    let salary = world.resource::<GameConfig>().salary;
    let pid = player_id_from_entity(world, entity);
    if let Some(mut p) = world.get_mut::<Player>(entity) {
        p.receive(salary);
    }
    events.push(GameEvent::SalaryCollected {
        player: pid,
        amount: salary,
    });
}

// ---------------------------------------------------------------------------
// Jail
// ---------------------------------------------------------------------------

fn send_to_jail(
    world: &mut World,
    entity: Entity,
    events: &mut Vec<GameEvent>,
    reason: JailReason,
) {
    let pid = player_id_from_entity(world, entity);
    if let Some(mut p) = world.get_mut::<Player>(entity) {
        p.in_jail = true;
        p.jail_turns = 0;
        p.position = JAIL_SQUARE;
        p.doubles_count = 0;
    }
    events.push(GameEvent::PlayerJailed {
        player: pid,
        reason,
    });
}

fn release_from_jail(
    world: &mut World,
    entity: Entity,
    events: &mut Vec<GameEvent>,
    method: ReleaseMethod,
) {
    let pid = player_id_from_entity(world, entity);
    if let Some(mut p) = world.get_mut::<Player>(entity) {
        p.in_jail = false;
        p.jail_turns = 0;
    }
    events.push(GameEvent::PlayerReleasedFromJail {
        player: pid,
        method,
    });
}

// ---------------------------------------------------------------------------
// Property helpers (public API)
// ---------------------------------------------------------------------------

/// Check if a player owns all unmortgaged properties in a color group.
pub fn owns_complete_set(world: &World, entity: Entity, group: PropertyGroup) -> bool {
    let group_sq = group_squares(group);
    let squares = world.resource::<Board>().squares;
    group_sq.iter().all(|&sq| {
        let sq_entity = squares[sq as usize];
        world
            .get::<Owned>(sq_entity)
            .map(|o| o.owner == entity && !o.is_mortgaged)
            .unwrap_or(false)
    })
}

/// Count railroads owned by a player.
pub fn count_railroads(world: &World, entity: Entity) -> u8 {
    let squares = world.resource::<Board>().squares;
    [5u8, 15, 25, 35]
        .iter()
        .filter(|&&sq| {
            world
                .get::<Owned>(squares[sq as usize])
                .map(|o| o.owner == entity)
                .unwrap_or(false)
        })
        .count() as u8
}

/// Count utilities owned by a player.
pub fn count_utilities(world: &World, entity: Entity) -> u8 {
    let squares = world.resource::<Board>().squares;
    [12u8, 28]
        .iter()
        .filter(|&&sq| {
            world
                .get::<Owned>(squares[sq as usize])
                .map(|o| o.owner == entity)
                .unwrap_or(false)
        })
        .count() as u8
}

/// Calculate rent for a square given dice and owner.
pub fn calculate_rent(world: &World, square: u8, dice: (u8, u8), owner: Entity) -> u32 {
    let squares = world.resource::<Board>().squares;
    let sq_entity = squares[square as usize];
    let owned = match world.get::<Owned>(sq_entity) {
        Some(o) => o,
        None => return 0,
    };
    if owned.is_mortgaged {
        return 0;
    }
    let kind = world
        .get::<BoardSquare>(sq_entity)
        .map(|bs| bs.kind)
        .unwrap_or(SquareKind::Go);

    match kind {
        SquareKind::Railroad => {
            let count = count_railroads(world, owner);
            25u32 << count.saturating_sub(1)
        }
        SquareKind::Utility => {
            let count = count_utilities(world, owner);
            (dice.0 as u32 + dice.1 as u32) * if count >= 2 { 10 } else { 4 }
        }
        SquareKind::Property(group) => {
            let prop = match world.get::<Property>(sq_entity) {
                Some(p) => p,
                None => return 0,
            };
            if owned.houses > 0 {
                let idx = (owned.houses.saturating_sub(1) as usize).min(4);
                prop.house_rent[idx]
            } else if owns_complete_set(world, owner, group) {
                prop.monopoly_rent
            } else {
                prop.base_rent
            }
        }
        _ => 0,
    }
}

/// Calculate total net worth (cash + properties + houses at half value).
pub fn calculate_net_worth(world: &World, entity: Entity) -> u32 {
    let cash = world.get::<Player>(entity).map(|p| p.cash).unwrap_or(0);
    let squares = world.resource::<Board>().squares;
    let mut value = 0u32;
    for &sq_entity in &squares {
        if let Some(owned) = world.get::<Owned>(sq_entity) {
            if owned.owner != entity {
                continue;
            }
            if owned.is_mortgaged {
                value += world
                    .get::<Property>(sq_entity)
                    .map(|p| p.mortgage_value)
                    .unwrap_or(0);
            } else if let Some(prop) = world.get::<Property>(sq_entity) {
                value += prop.price;
                value += (owned.houses.min(4) as u32) * prop.house_cost / 2;
            }
        }
    }
    cash + value
}

/// Check if a house can be built (even-building rule).
pub fn can_build_house(world: &World, entity: Entity, square: u8) -> bool {
    let squares = world.resource::<Board>().squares;
    let sq_entity = squares[square as usize];
    let group = match world.get::<BoardSquare>(sq_entity).map(|bs| bs.kind) {
        Some(SquareKind::Property(g)) => g,
        _ => return false,
    };
    let owned = match world.get::<Owned>(sq_entity) {
        Some(o) if o.owner == entity && !o.is_mortgaged => o,
        _ => return false,
    };
    if owned.houses > MAX_HOUSES {
        return false;
    }
    if !owns_complete_set(world, entity, group) {
        return false;
    }
    let min_houses = group_squares(group)
        .iter()
        .map(|&sq| {
            let e = squares[sq as usize];
            world
                .get::<Owned>(e)
                .filter(|o| o.owner == entity)
                .map(|o| o.houses)
                .unwrap_or(0)
        })
        .min()
        .unwrap_or(0);
    owned.houses <= min_houses + 1
}

/// Liquidate assets to raise cash. Returns total raised.
pub fn liquidate_assets(world: &mut World, entity: Entity, target: u32) -> u32 {
    let mut raised = 0u32;

    // Step 1: Sell houses (half price) — collect data first
    let sell_list: Vec<(u8, u8, u32)> = {
        let squares = world.resource::<Board>().squares;
        let mut list = Vec::new();
        for &sq_entity in &squares {
            if let Some(owned) = world.get::<Owned>(sq_entity)
                && owned.owner == entity
                && owned.houses > 0
            {
                let sq_idx = world
                    .get::<BoardSquare>(sq_entity)
                    .map(|bs| bs.index)
                    .unwrap_or(0);
                let cost = world
                    .get::<Property>(sq_entity)
                    .map(|p| p.house_cost)
                    .unwrap_or(0);
                list.push((sq_idx, owned.houses, cost));
            }
        }
        list
    };
    for (sq, houses, cost) in sell_list {
        if raised >= target {
            break;
        }
        let sq_entity = world.resource::<Board>().squares[sq as usize];
        let refund = (cost / 2) * houses as u32;
        if let Some(mut o) = world.get_mut::<Owned>(sq_entity) {
            o.houses = 0;
        }
        if let Some(mut p) = world.get_mut::<Player>(entity) {
            p.receive(refund);
        }
        raised += refund;
    }

    // Step 2: Mortgage unimproved properties — collect data first
    let mortgage_list: Vec<(u8, u32)> = {
        let squares = world.resource::<Board>().squares;
        let mut list = Vec::new();
        for &sq_entity in &squares {
            if let Some(owned) = world.get::<Owned>(sq_entity)
                && owned.owner == entity
                && !owned.is_mortgaged
                && owned.houses == 0
            {
                let sq_idx = world
                    .get::<BoardSquare>(sq_entity)
                    .map(|bs| bs.index)
                    .unwrap_or(0);
                let val = world
                    .get::<Property>(sq_entity)
                    .map(|p| p.mortgage_value)
                    .unwrap_or(0);
                list.push((sq_idx, val));
            }
        }
        list
    };
    for (sq, val) in mortgage_list {
        if raised >= target {
            break;
        }
        let sq_entity = world.resource::<Board>().squares[sq as usize];
        if let Some(mut o) = world.get_mut::<Owned>(sq_entity) {
            o.is_mortgaged = true;
        }
        if let Some(mut p) = world.get_mut::<Player>(entity) {
            p.receive(val);
        }
        raised += val;
    }

    raised
}

/// Transfer all assets from bankrupt player to creditor (or bank if None).
pub fn transfer_assets(world: &mut World, from: Entity, to: Option<Entity>) {
    let squares = world.resource::<Board>().squares;
    for &sq_entity in &squares {
        if let Some(mut owned) = world.get_mut::<Owned>(sq_entity)
            && owned.owner == from
        {
            if let Some(to_entity) = to {
                owned.owner = to_entity;
            } else {
                owned.owner = Entity::PLACEHOLDER;
                owned.is_mortgaged = false;
                owned.houses = 0;
            }
        }
    }
    if let Some(to_entity) = to {
        let cash = world.get::<Player>(from).map(|p| p.cash).unwrap_or(0);
        if let Some(mut r) = world.get_mut::<Player>(to_entity) {
            r.receive(cash);
        }
    }
    if let Some(mut p) = world.get_mut::<Player>(from) {
        p.cash = 0;
        p.is_bankrupt = true;
    }
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

fn draw_card(world: &mut World, is_chance: bool) -> CardEffect {
    let mut q = world.query::<&mut CardDeck>();
    for mut deck in q.iter_mut(world) {
        if deck.is_chance == is_chance {
            return deck.draw().clone();
        }
    }
    CardEffect::CollectMoney(0)
}

fn find_nearest(world: &World, from: u8, is_railroad: bool) -> u8 {
    let squares = world.resource::<Board>().squares;
    for offset in 1..=BOARD_SIZE {
        let pos = ((from as u16 + offset as u16) % BOARD_SIZE as u16) as u8;
        let kind = world
            .get::<BoardSquare>(squares[pos as usize])
            .map(|bs| bs.kind);
        let matches = if is_railroad {
            matches!(kind, Some(SquareKind::Railroad))
        } else {
            matches!(kind, Some(SquareKind::Utility))
        };
        if matches {
            return pos;
        }
    }
    from
}

fn execute_card_effect(
    world: &mut World,
    entity: Entity,
    effect: CardEffect,
    events: &mut Vec<GameEvent>,
) -> bool {
    let pid = player_id_from_entity(world, entity);

    match effect {
        CardEffect::CollectMoney(amount) => {
            if let Some(mut p) = world.get_mut::<Player>(entity) {
                p.receive(amount);
            }
        }
        CardEffect::PayMoney(amount) => {
            if !pay_debt(world, entity, amount, None, events) {
                return true;
            }
        }
        CardEffect::PayPerHouse { house, hotel } => {
            let squares = world.resource::<Board>().squares;
            let mut total = 0u32;
            for &sq_entity in &squares {
                if let Some(owned) = world.get::<Owned>(sq_entity)
                    && owned.owner == entity
                {
                    total += if owned.houses >= 5 {
                        hotel
                    } else {
                        owned.houses as u32 * house
                    };
                }
            }
            let _ = squares;
            if !pay_debt(world, entity, total, None, events) {
                return true;
            }
        }
        CardEffect::MoveTo(target) => {
            let old = world.get::<Player>(entity).map(|p| p.position).unwrap_or(0);
            if target < old {
                collect_salary(world, entity, events);
            }
            move_to(world, entity, target);
            events.push(GameEvent::PlayerMoved {
                player: pid,
                from: old,
                to: target,
                passed_go: target < old,
            });
        }
        CardEffect::MoveBack(spaces) => {
            let old = world.get::<Player>(entity).map(|p| p.position).unwrap_or(0);
            let new_pos = if old >= spaces {
                old - spaces
            } else {
                BOARD_SIZE - (spaces - old)
            };
            move_to(world, entity, new_pos);
            events.push(GameEvent::PlayerMoved {
                player: pid,
                from: old,
                to: new_pos,
                passed_go: false,
            });
        }
        CardEffect::MoveToNearest { is_railroad } => {
            let from = world.get::<Player>(entity).map(|p| p.position).unwrap_or(0);
            let target = find_nearest(world, from, is_railroad);
            if target < from {
                collect_salary(world, entity, events);
            }
            move_to(world, entity, target);
            events.push(GameEvent::PlayerMoved {
                player: pid,
                from,
                to: target,
                passed_go: target < from,
            });
        }
        CardEffect::GoToJail => {
            send_to_jail(world, entity, events, JailReason::CardEffect);
            return true;
        }
        CardEffect::GetOutOfJailFree => {
            if let Some(mut p) = world.get_mut::<Player>(entity) {
                p.get_out_of_jail_free += 1;
            }
        }
        CardEffect::PayEachPlayer(amount) => {
            let player_entities = world.resource::<PlayerEntities>().entities;
            let active = count_active_players(world).saturating_sub(1);
            let total = amount * active as u32;
            if !pay_debt(world, entity, total, None, events) {
                return true;
            }
            for (i, &other) in player_entities.iter().enumerate() {
                if other != entity
                    && is_player_active(world, i as u8)
                    && let Some(mut o) = world.get_mut::<Player>(other)
                {
                    o.receive(amount);
                }
            }
        }
        CardEffect::CollectFromEachPlayer(amount) => {
            let player_entities = world.resource::<PlayerEntities>().entities;
            for (i, &other) in player_entities.iter().enumerate() {
                if other != entity && is_player_active(world, i as u8) {
                    let can_pay = world
                        .get::<Player>(other)
                        .map(|p| p.cash >= amount)
                        .unwrap_or(false);
                    if can_pay {
                        if let Some(mut o) = world.get_mut::<Player>(other) {
                            o.pay(amount);
                        }
                        if let Some(mut p) = world.get_mut::<Player>(entity) {
                            p.receive(amount);
                        }
                    }
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Debt payment
// ---------------------------------------------------------------------------

fn pay_debt(
    world: &mut World,
    entity: Entity,
    amount: u32,
    creditor: Option<Entity>,
    events: &mut Vec<GameEvent>,
) -> bool {
    let cash = world.get::<Player>(entity).map(|p| p.cash).unwrap_or(0);
    if cash >= amount {
        if let Some(mut p) = world.get_mut::<Player>(entity) {
            p.pay(amount);
        }
        if let Some(c) = creditor
            && let Some(mut r) = world.get_mut::<Player>(c)
        {
            r.receive(amount);
        }
        return true;
    }
    let shortfall = amount - cash;
    liquidate_assets(world, entity, shortfall);
    let new_cash = world.get::<Player>(entity).map(|p| p.cash).unwrap_or(0);
    if new_cash >= amount {
        if let Some(mut p) = world.get_mut::<Player>(entity) {
            p.pay(amount);
        }
        if let Some(c) = creditor
            && let Some(mut r) = world.get_mut::<Player>(c)
        {
            r.receive(amount);
        }
        true
    } else {
        let pid = player_id_from_entity(world, entity);
        let cid = creditor.map(|e| player_id_from_entity(world, e));
        transfer_assets(world, entity, creditor);
        events.push(GameEvent::PlayerBankrupt {
            player: pid,
            creditor: cid,
        });
        false
    }
}

// ---------------------------------------------------------------------------
// Turn execution
// ---------------------------------------------------------------------------

/// Execute a single player's turn following the FSM phase sequence.
pub fn execute_turn(
    world: &mut World,
    player_id: u8,
    ai: &mut dyn MonopolyPlayer,
    rng: &mut fastrand::Rng,
) -> TurnResult {
    let mut events = Vec::new();
    let entity = player_entity(world, player_id);
    events.push(GameEvent::TurnStarted { player: player_id });

    // ── Phase 1: PreTurn (Jail) ──
    let in_jail = world
        .get::<Player>(entity)
        .map(|p| p.in_jail)
        .unwrap_or(false);

    if in_jail {
        let turn_number = world.resource::<TurnState>().turn_number;
        let ctx = build_ctx(world, player_id, turn_number);
        let decision = ai.jail_decision(&ctx);

        match decision {
            JailDecision::PayFine => {
                let fine = world.resource::<GameConfig>().jail_fine;
                if let Some(mut p) = world.get_mut::<Player>(entity) {
                    p.pay(fine);
                }
                release_from_jail(world, entity, &mut events, ReleaseMethod::PaidFine);
            }
            JailDecision::UseCard => {
                let has_card = world
                    .get::<Player>(entity)
                    .map(|p| p.get_out_of_jail_free > 0)
                    .unwrap_or(false);
                if has_card {
                    if let Some(mut p) = world.get_mut::<Player>(entity) {
                        p.get_out_of_jail_free -= 1;
                    }
                    release_from_jail(world, entity, &mut events, ReleaseMethod::UsedCard);
                }
            }
            JailDecision::RollForDoubles => {
                let (d1, d2, is_doubles) = roll_dice(rng);
                events.push(GameEvent::DiceRolled {
                    player: player_id,
                    die1: d1,
                    die2: d2,
                    doubles: is_doubles,
                });
                if is_doubles {
                    release_from_jail(world, entity, &mut events, ReleaseMethod::RolledDoubles);
                    let (new_pos, passed_go) = move_forward(world, entity, d1 + d2);
                    if passed_go {
                        collect_salary(world, entity, &mut events);
                    }
                    events.push(GameEvent::PlayerMoved {
                        player: player_id,
                        from: JAIL_SQUARE,
                        to: new_pos,
                        passed_go,
                    });
                    resolve_landing(
                        world,
                        entity,
                        player_id,
                        new_pos,
                        (d1, d2),
                        ai,
                        rng,
                        &mut events,
                    );
                } else {
                    if let Some(mut p) = world.get_mut::<Player>(entity) {
                        p.jail_turns += 1;
                        if p.jail_turns >= MAX_JAIL_TURNS {
                            release_from_jail(
                                world,
                                entity,
                                &mut events,
                                ReleaseMethod::MaxTurnsExceeded,
                            );
                        }
                    }
                    let bankrupt = world
                        .get::<Player>(entity)
                        .map(|p| p.is_bankrupt)
                        .unwrap_or(false);
                    return TurnResult {
                        player: player_id,
                        events,
                        went_bankrupt: bankrupt,
                    };
                }
            }
        }
    }

    if !is_player_active(world, player_id) {
        return TurnResult {
            player: player_id,
            events,
            went_bankrupt: true,
        };
    }

    // ── Phase 2-4: Rolling / Resolving / Doubles Loop ──
    let mut doubles_count = 0u8;

    loop {
        let (d1, d2, is_doubles) = roll_dice(rng);
        let dice = (d1, d2);
        events.push(GameEvent::DiceRolled {
            player: player_id,
            die1: d1,
            die2: d2,
            doubles: is_doubles,
        });

        if is_doubles {
            doubles_count += 1;
            if doubles_count >= MAX_DOUBLES {
                send_to_jail(world, entity, &mut events, JailReason::Speeding);
                break;
            }
        } else {
            doubles_count = 0;
        }

        let old_pos = world.get::<Player>(entity).map(|p| p.position).unwrap_or(0);
        let (new_pos, passed_go) = move_forward(world, entity, d1 + d2);
        if passed_go {
            collect_salary(world, entity, &mut events);
        }
        events.push(GameEvent::PlayerMoved {
            player: player_id,
            from: old_pos,
            to: new_pos,
            passed_go,
        });

        resolve_landing(
            world,
            entity,
            player_id,
            new_pos,
            dice,
            ai,
            rng,
            &mut events,
        );

        if !is_player_active(world, player_id) {
            break;
        }

        if is_doubles {
            let jailed = world
                .get::<Player>(entity)
                .map(|p| p.in_jail)
                .unwrap_or(false);
            if jailed {
                break;
            }
            continue;
        }
        break;
    }

    if !is_player_active(world, player_id) {
        return TurnResult {
            player: player_id,
            events,
            went_bankrupt: true,
        };
    }

    // ── Phase 5: Strategic (Build) ──
    let jailed = world
        .get::<Player>(entity)
        .map(|p| p.in_jail)
        .unwrap_or(false);
    if !jailed {
        let turn_number = world.resource::<TurnState>().turn_number;
        let ctx = build_ctx(world, player_id, turn_number);
        let builds = ai.build_houses(&ctx);

        for sq in builds {
            if can_build_house(world, entity, sq) {
                let squares = world.resource::<Board>().squares;
                let sq_entity = squares[sq as usize];
                let cost = world
                    .get::<Property>(sq_entity)
                    .map(|p| p.house_cost)
                    .unwrap_or(0);
                let can_afford = world
                    .get::<Player>(entity)
                    .map(|p| p.cash >= cost)
                    .unwrap_or(false);
                if can_afford {
                    if let Some(mut p) = world.get_mut::<Player>(entity) {
                        p.pay(cost);
                    }
                    if let Some(mut o) = world.get_mut::<Owned>(sq_entity) {
                        o.houses += 1;
                        events.push(GameEvent::HouseBuilt {
                            player: player_id,
                            square: sq,
                            houses: o.houses,
                        });
                    }
                    if let Some(mut stats) = world.get_resource_mut::<Statistics>() {
                        stats.record_house(player_id);
                    }
                }
            }
        }
    }

    if let Some(mut stats) = world.get_resource_mut::<Statistics>() {
        stats.turns_played += 1;
    }

    let bankrupt = world
        .get::<Player>(entity)
        .map(|p| p.is_bankrupt)
        .unwrap_or(false);
    TurnResult {
        player: player_id,
        events,
        went_bankrupt: bankrupt,
    }
}

// ---------------------------------------------------------------------------
// Square resolution
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn resolve_landing(
    world: &mut World,
    entity: Entity,
    player_id: u8,
    square: u8,
    dice: (u8, u8),
    ai: &mut dyn MonopolyPlayer,
    rng: &mut fastrand::Rng,
    events: &mut Vec<GameEvent>,
) {
    let squares = world.resource::<Board>().squares;
    let sq_entity = squares[square as usize];
    let kind = world
        .get::<BoardSquare>(sq_entity)
        .map(|bs| bs.kind)
        .unwrap_or(SquareKind::Go);

    match kind {
        SquareKind::Go | SquareKind::FreeParking | SquareKind::Jail => {}
        SquareKind::GoToJail => {
            send_to_jail(world, entity, events, JailReason::LandedOnGoToJail);
        }
        SquareKind::Tax(tax_kind) => {
            let amount = match tax_kind {
                TaxKind::Income => 200,
                TaxKind::Luxury => 100,
            };
            events.push(GameEvent::TaxPaid {
                player: player_id,
                amount,
                tax_kind,
            });
            pay_debt(world, entity, amount, None, events);
        }
        SquareKind::Chance => {
            let effect = draw_card(world, true);
            events.push(GameEvent::CardDrawn {
                player: player_id,
                is_chance: true,
                effect: effect.clone(),
            });
            let jailed = execute_card_effect(world, entity, effect, events);
            if !jailed {
                resolve_card_move(world, entity, player_id, square, events);
            }
        }
        SquareKind::CommunityChest => {
            let effect = draw_card(world, false);
            events.push(GameEvent::CardDrawn {
                player: player_id,
                is_chance: false,
                effect: effect.clone(),
            });
            let jailed = execute_card_effect(world, entity, effect, events);
            if !jailed {
                resolve_card_move(world, entity, player_id, square, events);
            }
        }
        SquareKind::Property(_) | SquareKind::Railroad | SquareKind::Utility => {
            resolve_property(
                world, entity, player_id, sq_entity, square, dice, ai, rng, events,
            );
        }
    }
}

fn resolve_card_move(
    world: &mut World,
    entity: Entity,
    player_id: u8,
    original_square: u8,
    events: &mut Vec<GameEvent>,
) {
    let new_pos = world
        .get::<Player>(entity)
        .map(|p| p.position)
        .unwrap_or(original_square);
    if new_pos == original_square {
        return;
    }
    let squares = world.resource::<Board>().squares;
    let sq_entity = squares[new_pos as usize];
    let kind = world
        .get::<BoardSquare>(sq_entity)
        .map(|bs| bs.kind)
        .unwrap_or(SquareKind::Go);
    match kind {
        SquareKind::GoToJail => {
            send_to_jail(world, entity, events, JailReason::CardEffect);
        }
        SquareKind::Tax(tax_kind) => {
            let amount = match tax_kind {
                TaxKind::Income => 200,
                TaxKind::Luxury => 100,
            };
            events.push(GameEvent::TaxPaid {
                player: player_id,
                amount,
                tax_kind,
            });
            pay_debt(world, entity, amount, None, events);
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_property(
    world: &mut World,
    entity: Entity,
    player_id: u8,
    sq_entity: Entity,
    square: u8,
    dice: (u8, u8),
    ai: &mut dyn MonopolyPlayer,
    rng: &mut fastrand::Rng,
    events: &mut Vec<GameEvent>,
) {
    let owned_opt = world.get::<Owned>(sq_entity);
    let is_owned = owned_opt.is_some();
    let is_self = owned_opt.map(|o| o.owner == entity).unwrap_or(false);
    let owner_entity = owned_opt.map(|o| o.owner);
    let _ = owned_opt;

    if !is_owned {
        // Unowned — ask AI to buy or auction
        let price = world
            .get::<Property>(sq_entity)
            .map(|p| p.price)
            .unwrap_or(0);
        let turn_number = world.resource::<TurnState>().turn_number;
        let ctx = build_ctx(world, player_id, turn_number);
        if ai.should_buy_property(&ctx, square, price) && ctx.cash >= price {
            if let Some(mut p) = world.get_mut::<Player>(entity) {
                p.pay(price);
            }
            world.entity_mut(sq_entity).insert(Owned::new(entity));
            events.push(GameEvent::PropertyBought {
                player: player_id,
                square,
                price,
            });
            if let Some(mut stats) = world.get_resource_mut::<Statistics>() {
                stats.record_purchase(player_id);
            }
        } else {
            events.push(GameEvent::PropertyDeclined {
                player: player_id,
                square,
            });
            run_auction(world, square, sq_entity, price, ai, rng, events);
        }
    } else if is_self {
        // Own property — nothing
    } else if let Some(owner_e) = owner_entity {
        // Opponent's property — pay rent
        let owner_bankrupt = world
            .get::<Player>(owner_e)
            .map(|p| p.is_bankrupt)
            .unwrap_or(true);
        if owner_bankrupt {
            return;
        }
        let rent = calculate_rent(world, square, dice, owner_e);
        if rent > 0 {
            let owner_id = player_id_from_entity(world, owner_e);
            events.push(GameEvent::RentPaid {
                payer: player_id,
                payee: owner_id,
                amount: rent,
                square,
            });
            if let Some(mut stats) = world.get_resource_mut::<Statistics>() {
                stats.record_rent(player_id, rent);
            }
            pay_debt(world, entity, rent, Some(owner_e), events);
        }
    }
}

fn run_auction(
    world: &mut World,
    square: u8,
    sq_entity: Entity,
    base_price: u32,
    ai: &mut dyn MonopolyPlayer,
    _rng: &mut fastrand::Rng,
    events: &mut Vec<GameEvent>,
) {
    events.push(GameEvent::AuctionStarted { square });
    let player_entities = world.resource::<PlayerEntities>().entities;
    let mut highest_bid = 0u32;
    let mut highest_bidder: Option<u8> = None;
    let turn_number = world.resource::<TurnState>().turn_number;

    for i in 0..4u8 {
        if !is_player_active(world, i) {
            continue;
        }
        let ctx = build_ctx(world, i, turn_number);
        let bid = ai.auction_bid(&ctx, square, highest_bid);
        if bid > highest_bid && bid <= ctx.cash {
            highest_bid = bid;
            highest_bidder = Some(i);
            events.push(GameEvent::AuctionBid {
                player: i,
                amount: bid,
            });
        }
    }

    // No bids → sell at minimum to first active player
    if highest_bidder.is_none() || highest_bid == 0 {
        for i in 0..4u8 {
            if !is_player_active(world, i) {
                continue;
            }
            let bidder_entity = player_entities[i as usize];
            let min_bid = (base_price / 2).max(super::AUCTION_MIN_BID);
            let can_afford = world
                .get::<Player>(bidder_entity)
                .map(|p| p.cash >= min_bid)
                .unwrap_or(false);
            if can_afford {
                highest_bid = min_bid;
                highest_bidder = Some(i);
                break;
            }
        }
    }

    if let Some(winner_id) = highest_bidder {
        let winner_entity = player_entities[winner_id as usize];
        if let Some(mut p) = world.get_mut::<Player>(winner_entity) {
            p.pay(highest_bid);
        }
        world
            .entity_mut(sq_entity)
            .insert(Owned::new(winner_entity));
        events.push(GameEvent::AuctionWon {
            player: winner_id,
            square,
            amount: highest_bid,
        });
        if let Some(mut stats) = world.get_resource_mut::<Statistics>() {
            stats.record_purchase(winner_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Game runner
// ---------------------------------------------------------------------------

/// Run a complete Monopoly game with 4 AI players.
pub fn run_game(
    seed: u64,
    players: &mut [Box<dyn MonopolyPlayer>; 4],
    rng: &mut fastrand::Rng,
    max_turns: u32,
) -> GameResult {
    let mut world = init_world(seed);
    let mut all_events: Vec<GameEvent> = Vec::new();
    let mut turn = 0u32;
    let mut current = 0u8;

    for p in players.iter_mut() {
        p.reset();
    }

    let winner = loop {
        if turn >= max_turns {
            break find_richest(&world);
        }
        let active = count_active_players(&world);
        if active <= 1 {
            break find_winner(&world);
        }
        if !is_player_active(&world, current) {
            current = (current + 1) % 4;
            continue;
        }

        let result = execute_turn(&mut world, current, players[current as usize].as_mut(), rng);
        all_events.extend(result.events);

        if result.went_bankrupt {
            let active = count_active_players(&world);
            if active <= 1 {
                break find_winner(&world);
            }
        }
        turn += 1;
        current = (current + 1) % 4;
    };

    all_events.push(GameEvent::GameOver { winner });
    GameResult {
        winner,
        total_turns: turn,
        events: all_events,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::players::RandomPlayer;
    use super::*;

    fn test_world() -> World {
        init_world(42)
    }

    #[test]
    fn init_game_creates_valid_world() {
        let world = test_world();
        assert!(world.get_resource::<GameConfig>().is_some());
        assert!(world.get_resource::<TurnState>().is_some());
        assert!(world.get_resource::<Statistics>().is_some());
        assert!(world.get_resource::<Board>().is_some());
        assert!(world.get_resource::<PlayerEntities>().is_some());
    }

    #[test]
    fn spawn_players_correct_starting_cash() {
        let world = test_world();
        let pe = world.resource::<PlayerEntities>();
        for i in 0..4 {
            let p = world.get::<Player>(pe.entities[i]).unwrap();
            assert_eq!(p.cash, super::super::STARTING_CASH);
            assert_eq!(p.position, 0);
            assert!(!p.in_jail);
        }
    }

    #[test]
    fn move_forward_wraps() {
        let mut world = test_world();
        let entity = player_entity(&world, 0);
        if let Some(mut p) = world.get_mut::<Player>(entity) {
            p.position = 38;
        }
        let (new_pos, passed_go) = move_forward(&mut world, entity, 5);
        assert_eq!(new_pos, 3);
        assert!(passed_go);
    }

    #[test]
    fn owns_complete_set_detection() {
        let mut world = test_world();
        let entity = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        assert!(!owns_complete_set(&world, entity, PropertyGroup::Brown));
        for sq in [1u8, 3] {
            world
                .entity_mut(squares[sq as usize])
                .insert(Owned::new(entity));
        }
        assert!(owns_complete_set(&world, entity, PropertyGroup::Brown));
    }

    #[test]
    fn rent_street_and_monopoly() {
        let mut world = test_world();
        let owner = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        world.entity_mut(squares[1]).insert(Owned::new(owner));
        assert_eq!(calculate_rent(&world, 1, (3, 4), owner), 2);
        world.entity_mut(squares[3]).insert(Owned::new(owner));
        assert_eq!(calculate_rent(&world, 1, (3, 4), owner), 4);
    }

    #[test]
    fn rent_railroad_scales() {
        let mut world = test_world();
        let owner = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        world.entity_mut(squares[5]).insert(Owned::new(owner));
        assert_eq!(calculate_rent(&world, 5, (1, 2), owner), 25);
        world.entity_mut(squares[15]).insert(Owned::new(owner));
        assert_eq!(calculate_rent(&world, 5, (1, 2), owner), 50);
    }

    #[test]
    fn rent_utility_multiplier() {
        let mut world = test_world();
        let owner = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        world.entity_mut(squares[12]).insert(Owned::new(owner));
        assert_eq!(calculate_rent(&world, 12, (3, 4), owner), 28);
        world.entity_mut(squares[28]).insert(Owned::new(owner));
        assert_eq!(calculate_rent(&world, 12, (3, 4), owner), 70);
    }

    #[test]
    fn rent_zero_mortgaged() {
        let mut world = test_world();
        let owner = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        world.entity_mut(squares[1]).insert(Owned {
            owner,
            is_mortgaged: true,
            houses: 0,
        });
        assert_eq!(calculate_rent(&world, 1, (3, 4), owner), 0);
    }

    #[test]
    fn can_build_even_rule() {
        let mut world = test_world();
        let entity = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        for sq in [6u8, 8, 9] {
            world
                .entity_mut(squares[sq as usize])
                .insert(Owned::new(entity));
        }
        assert!(can_build_house(&world, entity, 6));
    }

    #[test]
    fn liquidate_assets_works() {
        let mut world = test_world();
        let entity = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        world.entity_mut(squares[1]).insert(Owned::new(entity));
        if let Some(mut p) = world.get_mut::<Player>(entity) {
            p.cash = 0;
        }
        let raised = liquidate_assets(&mut world, entity, 30);
        assert!(raised >= 30);
        assert!(world.get::<Owned>(squares[1]).unwrap().is_mortgaged);
    }

    #[test]
    fn transfer_assets_bankruptcy() {
        let mut world = test_world();
        let loser = player_entity(&world, 0);
        let winner = player_entity(&world, 1);
        let squares = world.resource::<Board>().squares;
        world.entity_mut(squares[1]).insert(Owned::new(loser));
        transfer_assets(&mut world, loser, Some(winner));
        assert_eq!(world.get::<Owned>(squares[1]).unwrap().owner, winner);
        assert!(world.get::<Player>(loser).unwrap().is_bankrupt);
    }

    #[test]
    fn full_game_completes() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut players: [Box<dyn MonopolyPlayer>; 4] = [
            Box::new(RandomPlayer::new(0)),
            Box::new(RandomPlayer::new(1)),
            Box::new(RandomPlayer::new(2)),
            Box::new(RandomPlayer::new(3)),
        ];
        let result = run_game(42, &mut players, &mut rng, 1000);
        assert!(result.total_turns > 0);
        assert!(result.winner < 4);
        assert!(
            result
                .events
                .iter()
                .any(|e| matches!(e, GameEvent::GameOver { .. }))
        );
    }

    #[test]
    fn net_worth_includes_properties() {
        let mut world = test_world();
        let entity = player_entity(&world, 0);
        let squares = world.resource::<Board>().squares;
        world.entity_mut(squares[1]).insert(Owned::new(entity));
        assert_eq!(calculate_net_worth(&world, entity), 1560);
    }
}
