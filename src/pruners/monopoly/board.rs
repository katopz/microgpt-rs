use bevy_ecs::prelude::*;

use super::*;

/// Build the 40-square Monopoly board as entities in the ECS world.
/// Creates BoardSquare + Property/Railroad/Utility components for each square.
/// Inserts a `Board` resource mapping indices to entities.
pub fn build_board(world: &mut World) {
    let mut square_entities: Vec<Entity> = Vec::with_capacity(40);

    for i in 0..40u8 {
        let kind = square_kind(i);
        let entity = world.spawn(BoardSquare { index: i, kind }).id();

        // Add property data for street properties
        match kind {
            SquareKind::Property(group) => {
                let data = street_data(i, group);
                world.entity_mut(entity).insert(data);
            }
            SquareKind::Railroad => {
                world.entity_mut(entity).insert(Railroad);
                // Railroads have a simple property-like component
                world.entity_mut(entity).insert(Property {
                    square: i,
                    group: PropertyGroup::Brown, // placeholder, won't be used for railroads
                    name: square_name(i),
                    price: 200,
                    base_rent: 25,
                    monopoly_rent: 200, // max with 4 railroads
                    house_cost: 0,
                    house_rent: [0; 5],
                    mortgage_value: 100,
                });
            }
            SquareKind::Utility => {
                world.entity_mut(entity).insert(Utility);
                world.entity_mut(entity).insert(Property {
                    square: i,
                    group: PropertyGroup::Brown,
                    name: square_name(i),
                    price: 150,
                    base_rent: 0, // calculated dynamically
                    monopoly_rent: 0,
                    house_cost: 0,
                    house_rent: [0; 5],
                    mortgage_value: 75,
                });
            }
            _ => {}
        }

        square_entities.push(entity);
    }

    // Build the Board resource
    let arr: [Entity; 40] = square_entities
        .try_into()
        .expect("exactly 40 board squares");
    world.insert_resource(Board { squares: arr });

    // Create and shuffle card decks
    // Note: shuffle_decks is called separately with a seed
}

/// Returns Property component data for each street based on its square index.
fn street_data(square: u8, group: PropertyGroup) -> Property {
    match square {
        // Brown (2 properties)
        1 => Property {
            square: 1,
            group,
            name: "Mediterranean Ave",
            price: 60,
            base_rent: 2,
            monopoly_rent: 4,
            house_cost: 50,
            house_rent: [10, 30, 90, 160, 250],
            mortgage_value: 30,
        },
        3 => Property {
            square: 3,
            group,
            name: "Baltic Ave",
            price: 60,
            base_rent: 4,
            monopoly_rent: 8,
            house_cost: 50,
            house_rent: [20, 60, 180, 320, 450],
            mortgage_value: 30,
        },
        // Light Blue (3 properties)
        6 => Property {
            square: 6,
            group,
            name: "Oriental Ave",
            price: 100,
            base_rent: 6,
            monopoly_rent: 12,
            house_cost: 50,
            house_rent: [30, 90, 270, 400, 550],
            mortgage_value: 50,
        },
        8 => Property {
            square: 8,
            group,
            name: "Vermont Ave",
            price: 100,
            base_rent: 6,
            monopoly_rent: 12,
            house_cost: 50,
            house_rent: [30, 90, 270, 400, 550],
            mortgage_value: 50,
        },
        9 => Property {
            square: 9,
            group,
            name: "Connecticut Ave",
            price: 120,
            base_rent: 8,
            monopoly_rent: 16,
            house_cost: 50,
            house_rent: [40, 120, 360, 560, 750],
            mortgage_value: 60,
        },
        // Pink (3 properties)
        11 => Property {
            square: 11,
            group,
            name: "St. Charles Place",
            price: 140,
            base_rent: 10,
            monopoly_rent: 20,
            house_cost: 100,
            house_rent: [50, 150, 450, 625, 750],
            mortgage_value: 70,
        },
        13 => Property {
            square: 13,
            group,
            name: "States Ave",
            price: 140,
            base_rent: 10,
            monopoly_rent: 20,
            house_cost: 100,
            house_rent: [50, 150, 450, 625, 750],
            mortgage_value: 70,
        },
        14 => Property {
            square: 14,
            group,
            name: "Virginia Ave",
            price: 160,
            base_rent: 12,
            monopoly_rent: 24,
            house_cost: 100,
            house_rent: [60, 180, 500, 700, 900],
            mortgage_value: 80,
        },
        // Orange (3 properties)
        16 => Property {
            square: 16,
            group,
            name: "St. James Place",
            price: 180,
            base_rent: 14,
            monopoly_rent: 28,
            house_cost: 100,
            house_rent: [70, 200, 550, 750, 950],
            mortgage_value: 90,
        },
        18 => Property {
            square: 18,
            group,
            name: "Tennessee Ave",
            price: 180,
            base_rent: 14,
            monopoly_rent: 28,
            house_cost: 100,
            house_rent: [70, 200, 550, 750, 950],
            mortgage_value: 90,
        },
        19 => Property {
            square: 19,
            group,
            name: "New York Ave",
            price: 200,
            base_rent: 16,
            monopoly_rent: 32,
            house_cost: 100,
            house_rent: [80, 220, 600, 800, 1000],
            mortgage_value: 100,
        },
        // Red (3 properties)
        21 => Property {
            square: 21,
            group,
            name: "Kentucky Ave",
            price: 220,
            base_rent: 18,
            monopoly_rent: 36,
            house_cost: 150,
            house_rent: [90, 250, 700, 875, 1050],
            mortgage_value: 110,
        },
        23 => Property {
            square: 23,
            group,
            name: "Indiana Ave",
            price: 220,
            base_rent: 18,
            monopoly_rent: 36,
            house_cost: 150,
            house_rent: [90, 250, 700, 875, 1050],
            mortgage_value: 110,
        },
        24 => Property {
            square: 24,
            group,
            name: "Illinois Ave",
            price: 240,
            base_rent: 20,
            monopoly_rent: 40,
            house_cost: 150,
            house_rent: [100, 300, 750, 925, 1100],
            mortgage_value: 120,
        },
        // Yellow (3 properties)
        26 => Property {
            square: 26,
            group,
            name: "Atlantic Ave",
            price: 260,
            base_rent: 22,
            monopoly_rent: 44,
            house_cost: 150,
            house_rent: [110, 330, 800, 975, 1150],
            mortgage_value: 130,
        },
        27 => Property {
            square: 27,
            group,
            name: "Ventnor Ave",
            price: 260,
            base_rent: 22,
            monopoly_rent: 44,
            house_cost: 150,
            house_rent: [110, 330, 800, 975, 1150],
            mortgage_value: 130,
        },
        29 => Property {
            square: 29,
            group,
            name: "Marvin Gardens",
            price: 280,
            base_rent: 24,
            monopoly_rent: 48,
            house_cost: 150,
            house_rent: [120, 360, 850, 1025, 1200],
            mortgage_value: 140,
        },
        // Green (3 properties)
        31 => Property {
            square: 31,
            group,
            name: "Pacific Ave",
            price: 300,
            base_rent: 26,
            monopoly_rent: 52,
            house_cost: 200,
            house_rent: [130, 390, 900, 1100, 1275],
            mortgage_value: 150,
        },
        32 => Property {
            square: 32,
            group,
            name: "North Carolina Ave",
            price: 300,
            base_rent: 26,
            monopoly_rent: 52,
            house_cost: 200,
            house_rent: [130, 390, 900, 1100, 1275],
            mortgage_value: 150,
        },
        34 => Property {
            square: 34,
            group,
            name: "Pennsylvania Ave",
            price: 320,
            base_rent: 28,
            monopoly_rent: 56,
            house_cost: 200,
            house_rent: [150, 450, 1000, 1200, 1400],
            mortgage_value: 160,
        },
        // Dark Blue (2 properties)
        37 => Property {
            square: 37,
            group,
            name: "Park Place",
            price: 350,
            base_rent: 35,
            monopoly_rent: 70,
            house_cost: 200,
            house_rent: [175, 500, 1100, 1300, 1500],
            mortgage_value: 175,
        },
        39 => Property {
            square: 39,
            group,
            name: "Boardwalk",
            price: 400,
            base_rent: 50,
            monopoly_rent: 100,
            house_cost: 200,
            house_rent: [200, 600, 1400, 1700, 2000],
            mortgage_value: 200,
        },
        _ => panic!("No street data for square {square}"),
    }
}

// ── Card Deck Definitions ──────────────────────────────────────

/// Create the classic 16 Chance cards.
fn chance_cards() -> Vec<CardEffect> {
    vec![
        CardEffect::MoveTo(0),                            // Advance to GO
        CardEffect::MoveTo(24),                           // Advance to Illinois Ave
        CardEffect::MoveTo(11),                           // Advance to St. Charles Place
        CardEffect::MoveToNearest { is_railroad: true },  // Advance to nearest railroad (pay 2x)
        CardEffect::MoveToNearest { is_railroad: true },  // Advance to nearest railroad (pay 2x)
        CardEffect::MoveToNearest { is_railroad: false }, // Advance to nearest utility
        CardEffect::CollectMoney(50),                     // Bank pays dividend
        CardEffect::GetOutOfJailFree,                     // Get out of jail free
        CardEffect::MoveBack(3),                          // Go back 3 spaces
        CardEffect::GoToJail,                             // Go to jail
        CardEffect::PayPerHouse {
            house: 25,
            hotel: 100,
        }, // General repairs
        CardEffect::PayMoney(15),                         // Pay poor tax
        CardEffect::MoveTo(5),                            // Advance to Reading Railroad
        CardEffect::MoveTo(39),                           // Advance to Boardwalk
        CardEffect::PayEachPlayer(50),                    // Pay each player $50 (chairman)
        CardEffect::CollectMoney(150),                    // Loan matures
    ]
}

/// Create the classic 16 Community Chest cards.
fn community_chest_cards() -> Vec<CardEffect> {
    vec![
        CardEffect::MoveTo(0),         // Advance to GO
        CardEffect::CollectMoney(200), // Bank error in your favor
        CardEffect::PayMoney(50),      // Doctor's fee
        CardEffect::CollectMoney(50),  // Sale of stock
        CardEffect::GetOutOfJailFree,  // Get out of jail free
        CardEffect::GoToJail,          // Go to jail
        CardEffect::CollectMoney(100), // Holiday fund matures
        CardEffect::CollectMoney(20),  // Income tax refund
        CardEffect::PayEachPlayer(50), // Pay hospital bills
        CardEffect::PayMoney(100),     // School fees
        CardEffect::CollectMoney(25),  // Consultancy fee
        CardEffect::PayPerHouse {
            house: 40,
            hotel: 115,
        }, // Street repairs
        CardEffect::CollectMoney(10),  // Beauty contest
        CardEffect::CollectMoney(100), // Inheritance
        CardEffect::CollectFromEachPlayer(50), // Birthday
        CardEffect::CollectMoney(50),  // Life insurance matures
    ]
}

/// Shuffle card decks using the given seed and add them to the world.
pub fn shuffle_decks(world: &mut World, seed: u64) {
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut chance = chance_cards();
    shuffle_vec(&mut chance, &mut rng);
    world.spawn(CardDeck {
        cards: chance,
        draw_index: 0,
        is_chance: true,
    });

    let mut chest = community_chest_cards();
    shuffle_vec(&mut chest, &mut rng);
    world.spawn(CardDeck {
        cards: chest,
        draw_index: 0,
        is_chance: false,
    });
}

/// Fisher-Yates shuffle using fastrand.
fn shuffle_vec<T>(vec: &mut [T], rng: &mut fastrand::Rng) {
    for i in (1..vec.len()).rev() {
        let j = rng.usize(0..=i);
        vec.swap(i, j);
    }
}

// ── Property Group Helpers ─────────────────────────────────────

/// Returns how many properties are in each color group.
pub const fn group_size(group: PropertyGroup) -> u8 {
    match group {
        PropertyGroup::Brown => 2,
        PropertyGroup::LightBlue => 3,
        PropertyGroup::Pink => 3,
        PropertyGroup::Orange => 3,
        PropertyGroup::Red => 3,
        PropertyGroup::Yellow => 3,
        PropertyGroup::Green => 3,
        PropertyGroup::DarkBlue => 2,
    }
}

/// Returns all square indices belonging to a property group.
pub fn group_squares(group: PropertyGroup) -> Vec<u8> {
    match group {
        PropertyGroup::Brown => vec![1, 3],
        PropertyGroup::LightBlue => vec![6, 8, 9],
        PropertyGroup::Pink => vec![11, 13, 14],
        PropertyGroup::Orange => vec![16, 18, 19],
        PropertyGroup::Red => vec![21, 23, 24],
        PropertyGroup::Yellow => vec![26, 27, 29],
        PropertyGroup::Green => vec![31, 32, 34],
        PropertyGroup::DarkBlue => vec![37, 39],
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a fresh world with the board.
    fn test_world() -> World {
        let mut world = World::new();
        build_board(&mut world);
        shuffle_decks(&mut world, 42);
        world
    }

    #[test]
    fn test_all_40_squares_created() {
        let world = test_world();
        let board = world.resource::<Board>();
        assert_eq!(board.squares.len(), 40);

        // Verify each square has a BoardSquare component
        for (i, &entity) in board.squares.iter().enumerate() {
            let sq = world.get::<BoardSquare>(entity).unwrap();
            assert_eq!(sq.index, i as u8);
        }
    }

    #[test]
    fn test_all_22_streets_have_property_data() {
        let world = test_world();
        let board = world.resource::<Board>();

        let street_squares: [u8; 22] = [
            1, 3, 6, 8, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24, 26, 27, 29, 31, 32, 34, 37, 39,
        ];

        for &sq_idx in &street_squares {
            let entity = board.squares[sq_idx as usize];
            let prop = world.get::<Property>(entity).unwrap();
            assert_eq!(prop.square, sq_idx, "Property at square {sq_idx} mismatch");
            assert!(
                !prop.name.is_empty(),
                "Property at square {sq_idx} has no name"
            );
            let sq = world.get::<BoardSquare>(entity).unwrap();
            assert!(
                matches!(sq.kind, SquareKind::Property(_)),
                "Square {sq_idx} should be a Property kind"
            );
        }
    }

    #[test]
    fn test_all_4_railroads_have_railroad_component() {
        let world = test_world();
        let board = world.resource::<Board>();

        // Railroads are at squares 5, 15, 25, 35
        let railroad_squares: [u8; 4] = [5, 15, 25, 35];

        for &sq_idx in &railroad_squares {
            let entity = board.squares[sq_idx as usize];
            let sq = world.get::<BoardSquare>(entity).unwrap();
            assert_eq!(sq.kind, SquareKind::Railroad);

            let railroad = world.get::<Railroad>(entity);
            assert!(
                railroad.is_some(),
                "Square {sq_idx} missing Railroad component"
            );

            let prop = world.get::<Property>(entity).unwrap();
            assert_eq!(prop.price, 200);
            assert_eq!(prop.mortgage_value, 100);
        }
    }

    #[test]
    fn test_both_utilities_have_utility_component() {
        let world = test_world();
        let board = world.resource::<Board>();

        // Utilities are at squares 12 (Electric Company) and 28 (Water Works)
        let utility_squares: [u8; 2] = [12, 28];

        for &sq_idx in &utility_squares {
            let entity = board.squares[sq_idx as usize];
            let sq = world.get::<BoardSquare>(entity).unwrap();
            assert_eq!(sq.kind, SquareKind::Utility);

            let utility = world.get::<Utility>(entity);
            assert!(
                utility.is_some(),
                "Square {sq_idx} missing Utility component"
            );

            let prop = world.get::<Property>(entity).unwrap();
            assert_eq!(prop.price, 150);
            assert_eq!(prop.mortgage_value, 75);
        }
    }

    #[test]
    fn test_deck_sizes_are_16() {
        assert_eq!(chance_cards().len(), 16);
        assert_eq!(community_chest_cards().len(), 16);
    }

    #[test]
    fn test_decks_spawned_in_world() {
        let mut world = test_world();

        let mut deck_query = world.query::<&CardDeck>();
        let decks: Vec<&CardDeck> = deck_query.iter(&world).collect();

        assert_eq!(decks.len(), 2, "Expected 2 card decks");

        let chance_deck = decks.iter().find(|d| d.is_chance);
        let chest_deck = decks.iter().find(|d| !d.is_chance);

        assert!(chance_deck.is_some(), "Missing Chance deck");
        assert!(chest_deck.is_some(), "Missing Community Chest deck");

        assert_eq!(chance_deck.unwrap().cards.len(), 16);
        assert_eq!(chest_deck.unwrap().cards.len(), 16);
    }

    #[test]
    fn test_group_size_and_group_squares_consistency() {
        let all_groups = [
            PropertyGroup::Brown,
            PropertyGroup::LightBlue,
            PropertyGroup::Pink,
            PropertyGroup::Orange,
            PropertyGroup::Red,
            PropertyGroup::Yellow,
            PropertyGroup::Green,
            PropertyGroup::DarkBlue,
        ];

        let mut total_properties = 0u8;

        for group in all_groups {
            let size = group_size(group);
            let squares = group_squares(group);
            assert_eq!(
                squares.len(),
                size as usize,
                "group_size mismatch for {group:?}"
            );
            total_properties += size;
        }

        assert_eq!(total_properties, 22, "Expected 22 total street properties");
    }

    #[test]
    fn test_group_squares_match_street_data() {
        let all_groups = [
            PropertyGroup::Brown,
            PropertyGroup::LightBlue,
            PropertyGroup::Pink,
            PropertyGroup::Orange,
            PropertyGroup::Red,
            PropertyGroup::Yellow,
            PropertyGroup::Green,
            PropertyGroup::DarkBlue,
        ];

        for group in all_groups {
            for sq_idx in group_squares(group) {
                // street_data should not panic for valid squares
                let prop = street_data(sq_idx, group);
                assert_eq!(prop.square, sq_idx);
                assert_eq!(prop.group, group);
            }
        }
    }

    #[test]
    fn test_shuffle_produces_different_order_with_different_seeds() {
        let mut world_a = World::new();
        shuffle_decks(&mut world_a, 1);

        let mut world_b = World::new();
        shuffle_decks(&mut world_b, 999);

        let mut query_a = world_a.query::<&CardDeck>();
        let mut query_b = world_b.query::<&CardDeck>();

        let deck_a: Vec<&CardDeck> = query_a.iter(&world_a).collect();
        let deck_b: Vec<&CardDeck> = query_b.iter(&world_b).collect();

        // At least one deck should be shuffled differently
        let chance_a = deck_a.iter().find(|d| d.is_chance).unwrap();
        let chance_b = deck_b.iter().find(|d| d.is_chance).unwrap();

        // Extremely unlikely both seeds produce identical order
        let same_order = chance_a
            .cards
            .iter()
            .zip(chance_b.cards.iter())
            .all(|(a, b)| format!("{a:?}") == format!("{b:?}"));

        assert!(
            !same_order,
            "Different seeds should produce different shuffle orders"
        );
    }

    #[test]
    fn test_street_data_panic_on_invalid_square() {
        let result = std::panic::catch_unwind(|| {
            street_data(0, PropertyGroup::Brown); // GO is not a street
        });
        assert!(
            result.is_err(),
            "street_data should panic for non-street square"
        );
    }

    #[test]
    fn test_property_prices_monotonically_increase_per_group() {
        let all_groups = [
            (PropertyGroup::Brown, 60u16),
            (PropertyGroup::LightBlue, 100),
            (PropertyGroup::Pink, 140),
            (PropertyGroup::Orange, 180),
            (PropertyGroup::Red, 220),
            (PropertyGroup::Yellow, 260),
            (PropertyGroup::Green, 300),
            (PropertyGroup::DarkBlue, 350),
        ];

        for (group, min_price) in all_groups {
            for sq_idx in group_squares(group) {
                let prop = street_data(sq_idx, group);
                let prop_name = prop.name;
                let prop_price = prop.price;
                assert!(
                    prop_price >= min_price as u32,
                    "{prop_name} price {prop_price} < {min_price} for {group:?}"
                );
            }
        }
    }

    #[test]
    fn test_house_rent_increases_with_houses() {
        let all_groups = [
            PropertyGroup::Brown,
            PropertyGroup::LightBlue,
            PropertyGroup::Pink,
            PropertyGroup::Orange,
            PropertyGroup::Red,
            PropertyGroup::Yellow,
            PropertyGroup::Green,
            PropertyGroup::DarkBlue,
        ];

        for group in all_groups {
            for sq_idx in group_squares(group) {
                let prop = street_data(sq_idx, group);
                // Each house level should increase rent
                for h in 0..4 {
                    assert!(
                        prop.house_rent[h + 1] > prop.house_rent[h],
                        "{}: house_rent[{}] ({}) should be > house_rent[{}] ({})",
                        prop.name,
                        h + 1,
                        prop.house_rent[h + 1],
                        h,
                        prop.house_rent[h]
                    );
                }
            }
        }
    }

    #[test]
    fn test_mortgage_value_is_half_price() {
        let all_groups = [
            PropertyGroup::Brown,
            PropertyGroup::LightBlue,
            PropertyGroup::Pink,
            PropertyGroup::Orange,
            PropertyGroup::Red,
            PropertyGroup::Yellow,
            PropertyGroup::Green,
            PropertyGroup::DarkBlue,
        ];

        for group in all_groups {
            for sq_idx in group_squares(group) {
                let prop = street_data(sq_idx, group);
                assert_eq!(
                    prop.mortgage_value,
                    prop.price / 2,
                    "{}: mortgage should be half price",
                    prop.name
                );
            }
        }
    }
}
