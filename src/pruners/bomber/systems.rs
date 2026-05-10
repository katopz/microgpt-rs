//! Bomberman HL Arena — ECS game systems (bevy_ecs 0.15, World-based).
//!
//! All systems operate on `&mut World` for tick-based game loop control.
//! Called in deterministic order by [`run_tick`]; no ECS schedule is used.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;

use super::arena::ArenaGrid;
use super::{
    Alive, BOMB_FUSE_TICKS, Blast, Bomb, BombCount, BombFuse, BombRange, BomberAction, Cell,
    DEFAULT_BLAST_RANGE, DEFAULT_MAX_BOMBS, DEFAULT_SPEED, GameEvent, GameRng, GridPos, Player,
    PlayerEntities, PowerUp, PowerUpKind, SPAWN_POSITIONS, ScoreBoard, Speed, TICK_LIMIT,
    TickCounter,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A bomb whose fuse expired and is ready to explode.
pub struct PendingExplosion {
    pub pos: (i32, i32),
    pub range: u32,
    /// Player entity that placed the bomb (for active-count tracking).
    pub owner: Entity,
}

/// Four cardinal directions used for blast propagation.
const DIRECTIONS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

// ---------------------------------------------------------------------------
// World initialisation
// ---------------------------------------------------------------------------

/// Create a fresh `World` with all resources required for one bomberman round.
pub fn init_world(seed: u64) -> World {
    let mut world = World::new();
    world.insert_resource(ArenaGrid::generate(seed));
    world.insert_resource(GameRng { seed });
    world.insert_resource(TickCounter::default());
    world.insert_resource(ScoreBoard::default());
    world.init_resource::<Events<GameEvent>>();
    world
}

/// Spawn 4 player entities at the corner spawn positions.
///
/// Returns the 4 player [`Entity`] ids and inserts a [`PlayerEntities`] resource.
pub fn spawn_players(world: &mut World) -> [Entity; 4] {
    let entities: Vec<Entity> = SPAWN_POSITIONS
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| {
            world
                .spawn((
                    Player { id: i as u8 },
                    GridPos { x, y },
                    BombCount {
                        max: DEFAULT_MAX_BOMBS,
                        active: 0,
                    },
                    BombRange {
                        cells: DEFAULT_BLAST_RANGE,
                    },
                    Speed {
                        cells_per_tick: DEFAULT_SPEED,
                    },
                    Alive,
                ))
                .id()
        })
        .collect();

    let [a, b, c, d] = entities.try_into().expect("exactly 4 spawn positions");
    world.insert_resource(PlayerEntities {
        entities: [a, b, c, d],
    });
    [a, b, c, d]
}

// ---------------------------------------------------------------------------
// Tick orchestration
// ---------------------------------------------------------------------------

/// Run one game tick. Returns `true` while the round continues, `false` when ended.
pub fn run_tick(world: &mut World, actions: [Option<BomberAction>; 4]) -> bool {
    let pending = tick_bomb_fuses(world);
    let blast_cells = process_explosions(world, pending);
    apply_movement(world, actions);
    place_bombs(world, actions);
    collect_powerups(world);
    cleanup_and_check(world, blast_cells)
}

// ---------------------------------------------------------------------------
// 1. Bomb fuse countdown
// ---------------------------------------------------------------------------

fn tick_bomb_fuses(world: &mut World) -> Vec<PendingExplosion> {
    let mut to_explode: Vec<(Entity, (i32, i32), u32, Entity)> = Vec::new();

    {
        let mut q = world.query::<(Entity, &mut BombFuse, &GridPos, &BombRange)>();
        for (entity, mut fuse, pos, range) in q.iter_mut(world) {
            let owner = fuse.owner;
            fuse.ticks_remaining = fuse.ticks_remaining.saturating_sub(1);
            if fuse.ticks_remaining == 0 {
                to_explode.push((entity, (pos.x, pos.y), range.cells, owner));
            }
        }
    }

    let mut result = Vec::with_capacity(to_explode.len());
    for (entity, pos, range, owner) in to_explode {
        world.entity_mut(entity).despawn();
        world.send_event(GameEvent::BombExploded { pos, range });
        result.push(PendingExplosion { pos, range, owner });
    }
    result
}

// ---------------------------------------------------------------------------
// 2. Blast propagation
// ---------------------------------------------------------------------------

fn process_explosions(world: &mut World, queue: Vec<PendingExplosion>) -> Vec<(i32, i32)> {
    if queue.is_empty() {
        return Vec::new();
    }

    // ── Snapshot current state (immutable reads) ────────────────────
    let bomb_map: HashMap<(i32, i32), (Entity, u32, Entity)> = {
        let mut q = world.query_filtered::<(Entity, &GridPos, &BombRange, &BombFuse), With<Bomb>>();
        q.iter(world)
            .map(|(e, p, r, f)| ((p.x, p.y), (e, r.cells, f.owner)))
            .collect()
    };

    let player_map: HashMap<(i32, i32), (u8, Entity)> = {
        let mut q = world.query_filtered::<(Entity, &Player, &GridPos), With<Alive>>();
        q.iter(world)
            .map(|(e, p, pos)| ((pos.x, pos.y), (p.id, e)))
            .collect()
    };

    // Map player entity → player id (for killer tracking)
    let player_id_map: HashMap<Entity, u8> = player_map
        .values()
        .copied()
        .map(|(id, e)| (e, id))
        .collect();

    // ── Compute blast propagation (no world mutation) ───────────────
    let mut blast_cells: Vec<(i32, i32)> = Vec::new();
    let mut walls_destroyed: HashSet<(i32, i32)> = HashSet::new();
    let mut powerups_revealed: Vec<(PowerUpKind, (i32, i32))> = Vec::new();
    let mut players_killed: Vec<(u8, Entity, Option<u8>)> = Vec::new();
    let mut bombs_to_despawn: Vec<Entity> = Vec::new();
    let mut owners_to_decrement: HashSet<Entity> = HashSet::new();
    let mut processed: HashSet<(i32, i32)> = HashSet::new();

    let mut explosion_queue: Vec<PendingExplosion> = queue;

    while let Some(exp) = explosion_queue.pop() {
        let killer_id = player_id_map.get(&exp.owner).copied();
        processed.insert(exp.pos);
        blast_cells.push(exp.pos);
        owners_to_decrement.insert(exp.owner);

        if let Some(&(pid, pe)) = player_map.get(&exp.pos) {
            players_killed.push((pid, pe, killer_id));
        }

        for (dx, dy) in DIRECTIONS {
            for dist in 1..=exp.range as i32 {
                let cx = exp.pos.0 + dx * dist;
                let cy = exp.pos.1 + dy * dist;
                let cell = world.resource::<ArenaGrid>().get(cx, cy);

                match cell {
                    Cell::FixedWall => break,
                    Cell::DestructibleWall => {
                        blast_cells.push((cx, cy));
                        walls_destroyed.insert((cx, cy));
                        if let Some(&(pid, pe)) = player_map.get(&(cx, cy)) {
                            players_killed.push((pid, pe, killer_id));
                        }
                        break;
                    }
                    Cell::PowerUpHidden(kind) => {
                        blast_cells.push((cx, cy));
                        walls_destroyed.insert((cx, cy));
                        powerups_revealed.push((kind, (cx, cy)));
                        if let Some(&(pid, pe)) = player_map.get(&(cx, cy)) {
                            players_killed.push((pid, pe, killer_id));
                        }
                        break;
                    }
                    Cell::Floor => {
                        blast_cells.push((cx, cy));
                        if let Some(&(pid, pe)) = player_map.get(&(cx, cy)) {
                            players_killed.push((pid, pe, killer_id));
                        }
                        if processed.insert((cx, cy))
                            && let Some(&(be, br, bo)) = bomb_map.get(&(cx, cy))
                        {
                            bombs_to_despawn.push(be);
                            owners_to_decrement.insert(bo);
                            explosion_queue.push(PendingExplosion {
                                pos: (cx, cy),
                                range: br,
                                owner: bo,
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Apply mutations ─────────────────────────────────────────────
    {
        let mut grid = world.resource_mut::<ArenaGrid>();
        for &(x, y) in &walls_destroyed {
            grid.set(x, y, Cell::Floor);
        }
    }
    for &(x, y) in &walls_destroyed {
        world.send_event(GameEvent::WallDestroyed { pos: (x, y) });
    }

    for (kind, (x, y)) in powerups_revealed {
        world.spawn((PowerUp { kind }, GridPos { x, y }));
    }

    for be in bombs_to_despawn {
        world.entity_mut(be).despawn();
    }

    let mut killed_entities: HashSet<Entity> = HashSet::new();
    for (pid, pe, killer) in players_killed {
        if killed_entities.insert(pe) && world.get::<Alive>(pe).is_some() {
            world.entity_mut(pe).remove::<Alive>();
            world.send_event(GameEvent::PlayerKilled {
                victim: pid,
                killer,
            });
        }
    }

    for owner in owners_to_decrement {
        if let Some(mut c) = world.get_mut::<BombCount>(owner) {
            c.active = c.active.saturating_sub(1);
        }
    }

    blast_cells
}

// ---------------------------------------------------------------------------
// 3. Movement
// ---------------------------------------------------------------------------

fn apply_movement(world: &mut World, actions: [Option<BomberAction>; 4]) {
    let bomb_pos: HashSet<(i32, i32)> = {
        let mut q = world.query_filtered::<&GridPos, With<Bomb>>();
        q.iter(world).map(|p| (p.x, p.y)).collect()
    };

    // Collect phase — immutable query
    #[allow(clippy::type_complexity)]
    let mut moves: Vec<(u8, Entity, (i32, i32), (i32, i32))> = Vec::new();
    {
        let mut q = world.query_filtered::<(Entity, &Player, &GridPos), With<Alive>>();
        for (entity, player, pos) in q.iter(world) {
            let action = match actions.get(player.id as usize).copied().flatten() {
                Some(a) => a,
                None => continue,
            };
            let (dx, dy) = match action {
                BomberAction::Up => (0, -1),
                BomberAction::Down => (0, 1),
                BomberAction::Left => (-1, 0),
                BomberAction::Right => (1, 0),
                BomberAction::Bomb | BomberAction::Wait => continue,
            };
            let tx = pos.x + dx;
            let ty = pos.y + dy;

            if !world.resource::<ArenaGrid>().is_walkable(tx, ty) {
                continue;
            }
            if bomb_pos.contains(&(tx, ty)) {
                continue;
            }
            moves.push((player.id, entity, (pos.x, pos.y), (tx, ty)));
        }
    }

    // Apply phase — write new positions
    for (pid, entity, from, to) in moves {
        if let Some(mut pos) = world.get_mut::<GridPos>(entity) {
            pos.x = to.0;
            pos.y = to.1;
        }
        world.send_event(GameEvent::PlayerMoved {
            player: pid,
            from,
            to,
        });
    }
}

// ---------------------------------------------------------------------------
// 4. Bomb placement
// ---------------------------------------------------------------------------

fn place_bombs(world: &mut World, actions: [Option<BomberAction>; 4]) {
    let bomb_pos: HashSet<(i32, i32)> = {
        let mut q = world.query_filtered::<&GridPos, With<Bomb>>();
        q.iter(world).map(|p| (p.x, p.y)).collect()
    };

    let mut to_place: Vec<(Entity, (i32, i32), u32)> = Vec::new();
    {
        let mut q = world
            .query_filtered::<(Entity, &Player, &GridPos, &BombCount, &BombRange), With<Alive>>();
        for (entity, player, pos, count, range) in q.iter(world) {
            match actions.get(player.id as usize).copied().flatten() {
                Some(BomberAction::Bomb) => {}
                _ => continue,
            }
            if count.active >= count.max {
                continue;
            }
            if bomb_pos.contains(&(pos.x, pos.y)) {
                continue;
            }
            to_place.push((entity, (pos.x, pos.y), range.cells));
        }
    }

    for (owner, (x, y), range) in to_place {
        world.spawn((
            Bomb,
            GridPos { x, y },
            BombFuse {
                owner,
                ticks_remaining: BOMB_FUSE_TICKS,
            },
            BombRange { cells: range },
        ));
        if let Some(mut c) = world.get_mut::<BombCount>(owner) {
            c.active += 1;
        }
        let pid = world.get::<Player>(owner).map(|p| p.id).unwrap_or(0);
        world.send_event(GameEvent::BombPlaced {
            player: pid,
            pos: (x, y),
        });
    }
}

// ---------------------------------------------------------------------------
// 5. Power-up collection
// ---------------------------------------------------------------------------

fn collect_powerups(world: &mut World) {
    let pu_map: HashMap<(i32, i32), (Entity, PowerUpKind)> = {
        let mut q = world.query_filtered::<(Entity, &PowerUp, &GridPos), ()>();
        q.iter(world)
            .map(|(e, pu, pos)| ((pos.x, pos.y), (e, pu.kind)))
            .collect()
    };

    if pu_map.is_empty() {
        return;
    }

    let mut to_collect: Vec<(Entity, Entity, PowerUpKind)> = Vec::new();
    {
        let mut pq = world.query_filtered::<(Entity, &Player, &GridPos), With<Alive>>();
        for (pe, _player, pos) in pq.iter(world) {
            if let Some(&(pue, kind)) = pu_map.get(&(pos.x, pos.y)) {
                to_collect.push((pe, pue, kind));
            }
        }
    }

    for (player_entity, pu_entity, kind) in to_collect {
        match kind {
            PowerUpKind::BombUp => {
                if let Some(mut c) = world.get_mut::<BombCount>(player_entity) {
                    c.max += 1;
                }
            }
            PowerUpKind::FireUp => {
                if let Some(mut r) = world.get_mut::<BombRange>(player_entity) {
                    r.cells += 1;
                }
            }
            PowerUpKind::SpeedUp => {
                if let Some(mut s) = world.get_mut::<Speed>(player_entity) {
                    s.cells_per_tick = (s.cells_per_tick + 1).min(2);
                }
            }
        }
        let pid = world
            .get::<Player>(player_entity)
            .map(|p| p.id)
            .unwrap_or(0);
        world.send_event(GameEvent::PowerUpCollected { player: pid, kind });
        world.entity_mut(pu_entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// 6. Cleanup + round-end check
// ---------------------------------------------------------------------------

fn cleanup_and_check(world: &mut World, blast_cells: Vec<(i32, i32)>) -> bool {
    // Remove previous tick's blast visuals
    {
        let old: Vec<Entity> = {
            let mut q = world.query_filtered::<Entity, With<Blast>>();
            q.iter(world).collect()
        };
        for e in old {
            world.entity_mut(e).despawn();
        }
    }

    // Spawn blast visuals for this tick
    for (x, y) in blast_cells {
        world.spawn((Blast, GridPos { x, y }));
    }

    // Advance tick counter
    world.resource_mut::<TickCounter>().tick += 1;

    // Count survivors
    let alive: Vec<u8> = {
        let mut q = world.query_filtered::<(Entity, &Player), With<Alive>>();
        q.iter(world).map(|(_, p)| p.id).collect()
    };

    let tick = world.resource::<TickCounter>().tick;
    let round_over = alive.len() <= 1 || tick >= TICK_LIMIT;

    if round_over {
        world.send_event(GameEvent::RoundEnd { survivors: alive });
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_world_creates_required_resources() {
        let world = init_world(42);
        assert!(world.contains_resource::<ArenaGrid>());
        assert!(world.contains_resource::<TickCounter>());
        assert!(world.contains_resource::<ScoreBoard>());
        assert!(world.contains_resource::<Events<GameEvent>>());
    }

    #[test]
    fn spawn_players_creates_four_entities() {
        let mut world = init_world(42);
        let entities = spawn_players(&mut world);
        assert_eq!(entities.len(), 4);

        // Verify PlayerEntities resource was inserted
        let pe = world.resource::<PlayerEntities>();
        assert_eq!(pe.entities, entities);

        // Verify each player has correct id and spawn position
        for (i, &entity) in entities.iter().enumerate() {
            let player = world.get::<Player>(entity).unwrap();
            assert_eq!(player.id, i as u8);
            let pos = world.get::<GridPos>(entity).unwrap();
            assert_eq!((pos.x, pos.y), SPAWN_POSITIONS[i]);
        }
    }

    #[test]
    fn run_tick_advances_counter() {
        let mut world = init_world(42);
        spawn_players(&mut world);
        let before = world.resource::<TickCounter>().tick;
        let ongoing = run_tick(&mut world, [None; 4]);
        assert!(ongoing, "round should continue with no bombs");
        assert_eq!(world.resource::<TickCounter>().tick, before + 1);
    }

    #[test]
    fn bomb_explodes_after_fuse_ticks() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Place a bomb at player 0 position
        let pos = *world.get::<GridPos>(p0).unwrap();
        world.spawn((
            Bomb,
            GridPos { x: pos.x, y: pos.y },
            BombFuse {
                owner: p0,
                ticks_remaining: BOMB_FUSE_TICKS,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        // Track the owner's active count
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Run ticks until bomb explodes
        for _ in 0..BOMB_FUSE_TICKS {
            let _ = run_tick(&mut world, [None; 4]);
        }

        // Bomb should be gone, active count decremented
        assert_eq!(world.get_mut::<BombCount>(p0).unwrap().active, 0);

        // No bomb entities left
        let bomb_count = {
            let mut q = world.query_filtered::<Entity, With<Bomb>>();
            q.iter(&world).count()
        };
        assert_eq!(bomb_count, 0);
    }
}
