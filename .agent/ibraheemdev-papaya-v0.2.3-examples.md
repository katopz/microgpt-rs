# Code Examples for ibraheemdev-papaya (Version: v0.2.3)

## `doc_comment:src/lib.rs:1:0`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
use papaya::HashMap;

// Create a map.
let map = HashMap::new();

// Pin the map.
let map = map.pin();

// Use the map as normal.
map.insert('A', 1);
assert_eq!(map.get(&'A'), Some(&1));
assert_eq!(map.len(), 1);
```
---
## `doc_comment:src/lib.rs:1:1`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
use papaya::HashMap;

// Use a map from multiple threads.
let map = HashMap::new();
std::thread::scope(|s| {
// Insert some values.
s.spawn(|| {
let map = map.pin();
for i in 'A'..='Z' {
map.insert(i, 1);
}
});

// Remove the values.
s.spawn(|| {
let map = map.pin();
for i in 'A'..='Z' {
map.remove(&i);
}
});

// Read the values.
s.spawn(|| {
for (key, value) in map.pin().iter() {
println!("{key}: {value}");
}
});
});
```
---
## `doc_comment:src/lib.rs:1:2`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
let map = papaya::HashMap::new();
map.pin().insert("poneyland", 42);
assert_eq!(map.pin().update("poneyland", |e| e + 1), Some(&43));
```
---
## `doc_comment:src/lib.rs:1:3`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
use std::collections::HashMap;

let mut map = HashMap::new();
// Insert `poneyland` with the value `42` if it doesn't exist,
// otherwise increment it's value.
map.entry("poneyland")
.and_modify(|e| { *e += 1 })
.or_insert(42);
```
---
## `doc_comment:src/lib.rs:1:4`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
use papaya::HashMap;

let map = HashMap::new();
// Insert `poneyland` with the value `42` if it doesn't exist,
// otherwise increment it's value.
map.pin().update_or_insert("poneyland", |e| e + 1, 42);
```
---
## `doc_comment:src/lib.rs:1:5`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
# use std::sync::Arc;
use papaya::HashMap;

async fn run(map: Arc<HashMap<i32, String>>) {
tokio::spawn(async move {
// Pin the map with an owned guard.
let map = map.pin_owned();

// Hold references across await points.
let value = map.get(&37);
tokio::fs::write("db.txt", format!("{value:?}")).await;
println!("{value:?}");
});
}
```
---
## `doc_comment:src/lib.rs:1:6`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
# use std::sync::Arc;
use papaya::HashMap;

async fn run(map: Arc<HashMap<i32, String>>) {
tokio::spawn(async move {
for (key, value) in map.pin_owned().iter() {
tokio::fs::write("db.txt", format!("{key}: {value}\n")).await;
}
});
}
```
---
## `doc_comment:src/lib.rs:1:7`
**Source:** `src/lib.rs` (`doc_comment`)

```rust
use papaya::Guard;

pub struct Metrics {
map: papaya::HashMap<String, Vec<u64>>
}

impl Metrics {
pub fn guard(&self) -> impl Guard + '_ {
self.map.guard()
}

pub fn get<'guard>(&self, name: &str, guard: &'guard impl Guard) -> Option<&'guard [u64]> {
Some(self.map.get(name, guard)?.as_slice())
}
}
```
---
## `doc_comment:src/map.rs:37:0`
**Source:** `src/map.rs` (`doc_comment`)

```rust
use papaya::{HashMap, ResizeMode};
use seize::Collector;
use std::collections::hash_map::RandomState;

let map: HashMap<i32, i32> = HashMap::builder()
// Set the initial capacity.
.capacity(2048)
// Set the hasher.
.hasher(RandomState::new())
// Set the resize mode.
.resize_mode(ResizeMode::Blocking)
// Set a custom garbage collector.
.collector(Collector::new().batch_size(128))
// Construct the hash map.
.build();
```
---
## `doc_comment:src/map.rs:748:0`
**Source:** `src/map.rs` (`doc_comment`)

```rust
use papaya::{HashMap, Operation, Compute};

let map = HashMap::new();
let map = map.pin();

let compute = |entry| match entry {
// Remove the value if it is even.
Some((_key, value)) if value % 2 == 0 => {
Operation::Remove
}

// Increment the value if it is odd.
Some((_key, value)) => {
Operation::Insert(value + 1)
}

// Do nothing if the key does not exist
None => Operation::Abort(()),
};

assert_eq!(map.compute('A', compute), Compute::Aborted(()));

map.insert('A', 1);
assert_eq!(map.compute('A', compute), Compute::Updated {
old: (&'A', &1),
new: (&'A', &2),
});
assert_eq!(map.compute('A', compute), Compute::Removed(&'A', &2));
```
---
## `doc_comment:src/set.rs:26:0`
**Source:** `src/set.rs` (`doc_comment`)

```rust
use papaya::{HashSet, ResizeMode};
use seize::Collector;
use std::collections::hash_map::RandomState;

let set: HashSet<i32> = HashSet::builder()
// Set the initial capacity.
.capacity(2048)
// Set the hasher.
.hasher(RandomState::new())
// Set the resize mode.
.resize_mode(ResizeMode::Blocking)
// Set a custom garbage collector.
.collector(Collector::new().batch_size(128))
// Construct the hash set.
.build();
```
---
## `test:tests/basic.rs:clear`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        {
            map.insert(0, 1, &guard);
            map.insert(1, 1, &guard);
            map.insert(2, 1, &guard);
            map.insert(3, 1, &guard);
            map.insert(4, 1, &guard);
```
---
## `test:tests/basic.rs:clone_map_empty`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<&'static str, u32>(|map| {
        let map = map();
        let cloned_map = map.clone();
        assert_eq!(map.len(), cloned_map.len());
        assert_eq!(&map, &cloned_map);
        assert_eq!(cloned_map.len(), 0);
```
---
## `test:tests/basic.rs:compute`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let map = map.pin();

        for i in 0..100 {
            let compute = |entry| match entry {
                Some((_, value)) if value % 2 == 0 => Operation::Remove,
                Some((_, value)) => Operation::Insert(value + 1),
                None => Operation::Abort(()),
```
---
## `test:tests/basic.rs:concurrent_insert`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let map = Arc::new(map);

        let map1 = map.clone();
        let t1 = std::thread::spawn(move || {
            for i in 0..64 {
                map1.insert(i, 0, &map1.guard());
```
---
## `test:tests/basic.rs:concurrent_remove`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let map = Arc::new(map);

        {
            let guard = map.guard();
            for i in 0..64 {
                map.insert(i, i, &guard);
```
---
## `test:tests/basic.rs:current_kv_dropped`
**Source:** `tests/basic.rs` (`test`)

```rust
let dropped1 = Arc::new(0);
    let dropped2 = Arc::new(0);

    with_map::<Arc<usize>, Arc<usize>>(|map| {
        let map = map();
        map.insert(dropped1.clone(), dropped2.clone(), &map.guard());
        assert_eq!(Arc::strong_count(&dropped1), 2);
        assert_eq!(Arc::strong_count(&dropped2), 2);

        drop(map);

        // dropping the map should immediately drop (not deferred) all keys and values
        assert_eq!(Arc::strong_count(&dropped1), 1);
        assert_eq!(Arc::strong_count(&dropped2), 1);
```
---
## `test:tests/basic.rs:debug`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        map.insert(42, 0, &guard);
        map.insert(16, 8, &guard);

        let formatted = format!("{:?
```
---
## `test:tests/basic.rs:default`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        map.insert(42, 0, &guard);

        assert_eq!(map.get(&42, &guard), Some(&0));
```
---
## `test:tests/basic.rs:different_size_maps_not_equal`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map1| {
        let map1 = map1();
        with_map::<usize, usize>(|map2| {
            let map2 = map2();
            {
                let guard1 = map1.guard();
                let guard2 = map2.guard();

                map1.insert(1, 0, &guard1);
                map1.insert(2, 0, &guard1);
                map1.insert(3, 0, &guard1);

                map2.insert(1, 0, &guard2);
                map2.insert(2, 0, &guard2);
```
---
## `test:tests/basic.rs:different_values_not_equal`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map1| {
        let map1 = map1();
        with_map::<usize, usize>(|map2| {
            let map2 = map2();
            {
                map1.pin().insert(1, 0);
                map2.pin().insert(1, 1);
```
---
## `test:tests/basic.rs:empty_maps_equal`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map1| {
        let map1 = map1();
        with_map::<usize, usize>(|map2| {
            let map2 = map2();
            assert_eq!(map1, map2);
            assert_eq!(map2, map1);
```
---
## `test:tests/basic.rs:extend`
**Source:** `tests/basic.rs` (`test`)

```rust
if cfg!(papaya_stress) {
        return;
```
---
## `test:tests/basic.rs:from_iter_empty`
**Source:** `tests/basic.rs` (`test`)

```rust
use std::iter::FromIterator;

    let entries: Vec<(usize, usize)> = Vec::new();
    let map: HashMap<usize, usize> = HashMap::from_iter(entries.into_iter());

    assert_eq!(map.len(), 0)
```
---
## `test:tests/basic.rs:from_iter_repeated`
**Source:** `tests/basic.rs` (`test`)

```rust
use std::iter::FromIterator;

    let entries = vec![(0, 1), (0, 2), (0, 3)];
    let map: HashMap<_, _> = HashMap::from_iter(entries.into_iter());
    let map = map.pin();
    assert_eq!(map.len(), 1);
    assert_eq!(map.iter().collect::<Vec<_>>(), vec![(&0, &3)])
```
---
## `test:tests/basic.rs:get_empty`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        let e = map.get(&42, &guard);
        assert!(e.is_none());
```
---
## `test:tests/basic.rs:get_key_value_empty`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        let e = map.get_key_value(&42, &guard);
        assert!(e.is_none());
```
---
## `test:tests/basic.rs:get_or_insert`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();

        let result = map.get_or_insert(42, 0, &guard);
        assert_eq!(result, &0);
        assert_eq!(map.len(), 1);

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &0);
```
---
## `test:tests/basic.rs:get_or_insert_with`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();

        let result = map.get_or_insert_with(42, || 0, &guard);
        assert_eq!(result, &0);
        assert_eq!(map.len(), 1);

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &0);
```
---
## `test:tests/basic.rs:insert`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        let old = map.insert(42, 0, &guard);
        assert!(old.is_none());
```
---
## `test:tests/basic.rs:insert_and_get`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        map.insert(42, 0, &map.guard());

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &0);
```
---
## `test:tests/basic.rs:insert_and_get_key_value`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        map.insert(42, 0, &map.guard());

        {
            let guard = map.guard();
            let e = map.get_key_value(&42, &guard).unwrap();
            assert_eq!(e, (&42, &0));
```
---
## `test:tests/basic.rs:insert_and_remove`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        map.insert(42, 0, &guard);
        let old = map.remove(&42, &guard).unwrap();
        assert_eq!(old, &0);
        assert!(map.get(&42, &guard).is_none());
```
---
## `test:tests/basic.rs:len`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let len = if cfg!(miri) { 100
```
---
## `test:tests/basic.rs:mixed`
**Source:** `tests/basic.rs` (`test`)

```rust
const LEN: usize = if cfg!(miri) { 48
```
---
## `test:tests/basic.rs:new`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| drop(map()));
```
---
## `test:tests/basic.rs:reinsert`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        map.insert(42, 0, &guard);
        let old = map.insert(42, 1, &guard);
        assert_eq!(old, Some(&0));

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &1);
```
---
## `test:tests/basic.rs:remove_empty`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        let old = map.remove(&42, &guard);
        assert!(old.is_none());
```
---
## `test:tests/basic.rs:remove_if`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();

        assert_eq!(map.pin().remove_if(&1, |_k, _v| true), Ok(None));

        map.pin().insert(1, 0);

        assert_eq!(map.pin().remove_if(&0, |_k, _v| true), Ok(None));
        assert_eq!(map.pin().remove_if(&1, |_k, _v| false), Err((&1, &0)));
        assert_eq!(map.pin().remove_if(&1, |_k, v| *v == 1), Err((&1, &0)));

        assert_eq!(map.pin().get(&1), Some(&0));

        assert_eq!(map.pin().remove_if(&1, |_k, v| *v == 0), Ok(Some((&1, &0))));
        assert_eq!(map.pin().remove_if(&1, |_, _| true), Ok(None));
```
---
## `test:tests/basic.rs:retain_all_false`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        for i in 0..100 {
            map.pin().insert(i, i);
```
---
## `test:tests/basic.rs:retain_empty`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        map.pin().retain(|_, _| false);
        assert_eq!(map.len(), 0);
```
---
## `test:tests/basic.rs:same_values_equal`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map1| {
        let map1 = map1();
        with_map::<usize, usize>(|map2| {
            let map2 = map2();
            {
                map1.pin().insert(1, 0);
                map2.pin().insert(1, 0);
```
---
## `test:tests/basic.rs:test_max_hasher`
**Source:** `tests/basic.rs` (`test`)

```rust
#[derive(Default)]
        struct MaxHasher;

        impl Hasher for MaxHasher {
            fn finish(&self) -> u64 {
                u64::max_value()
```
---
## `test:tests/basic.rs:test_zero_hasher`
**Source:** `tests/basic.rs` (`test`)

```rust
#[derive(Default)]
        pub struct ZeroHasher;

        impl Hasher for ZeroHasher {
            fn finish(&self) -> u64 {
                0
```
---
## `test:tests/basic.rs:try_insert`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();

        assert_eq!(map.try_insert(42, 1, &guard), Ok(&1));
        assert_eq!(map.len(), 1);

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &1);
```
---
## `test:tests/basic.rs:try_insert_with`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();

        map.try_insert_with(42, || 1, &guard).unwrap();
        assert_eq!(map.len(), 1);

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &1);
```
---
## `test:tests/basic.rs:update`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        map.insert(42, 0, &guard);
        assert_eq!(map.len(), 1);
        let new = map.update(42, |v| v + 1, &guard);
        assert_eq!(map.len(), 1);
        assert_eq!(new, Some(&1));

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &1);
```
---
## `test:tests/basic.rs:update_empty`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();
        let new = map.update(42, |v| v + 1, &guard);
        assert!(new.is_none());

        {
            let guard = map.guard();
            assert!(map.get(&42, &guard).is_none());
```
---
## `test:tests/basic.rs:update_or_insert`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();

        let result = map.update_or_insert(42, |v| v + 1, 0, &guard);
        assert_eq!(result, &0);
        assert_eq!(map.len(), 1);

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &0);
```
---
## `test:tests/basic.rs:update_or_insert_with`
**Source:** `tests/basic.rs` (`test`)

```rust
with_map::<usize, usize>(|map| {
        let map = map();
        let guard = map.guard();

        let result = map.update_or_insert_with(42, |v| v + 1, || 0, &guard);
        assert_eq!(result, &0);
        assert_eq!(map.len(), 1);

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &0);
```
---
## `test:tests/basic_set.rs:clear`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        {
            set.insert(0, &guard);
            set.insert(1, &guard);
            set.insert(2, &guard);
            set.insert(3, &guard);
            set.insert(4, &guard);
```
---
## `test:tests/basic_set.rs:clone_set_empty`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<&'static str>(|set| {
        let set = set();
        let cloned_set = set.clone();
        assert_eq!(set.len(), cloned_set.len());
        assert_eq!(&set, &cloned_set);
        assert_eq!(cloned_set.len(), 0);
```
---
## `test:tests/basic_set.rs:concurrent_insert`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let set = Arc::new(set);

        let set1 = set.clone();
        let t1 = std::thread::spawn(move || {
            for i in 0..64 {
                set1.insert(i, &set1.guard());
```
---
## `test:tests/basic_set.rs:concurrent_remove`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let set = Arc::new(set);

        {
            let guard = set.guard();
            for i in 0..64 {
                set.insert(i, &guard);
```
---
## `test:tests/basic_set.rs:current_kv_dropped`
**Source:** `tests/basic_set.rs` (`test`)

```rust
let dropped1 = Arc::new(0);

    with_set::<Arc<usize>>(|set| {
        let set = set();
        set.insert(dropped1.clone(), &set.guard());
        assert_eq!(Arc::strong_count(&dropped1), 2);

        drop(set);

        // dropping the set should immediately drop (not deferred) all keys and values
        assert_eq!(Arc::strong_count(&dropped1), 1);
```
---
## `test:tests/basic_set.rs:debug`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        set.insert(42, &guard);
        set.insert(16, &guard);

        let formatted = format!("{:?
```
---
## `test:tests/basic_set.rs:default`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        set.insert(42, &guard);

        assert_eq!(set.get(&42, &guard), Some(&42));
```
---
## `test:tests/basic_set.rs:different_size_sets_not_equal`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            {
                let guard1 = set1.guard();
                let guard2 = set2.guard();

                set1.insert(1, &guard1);
                set1.insert(2, &guard1);
                set1.insert(3, &guard1);

                set2.insert(1, &guard2);
                set2.insert(2, &guard2);
```
---
## `test:tests/basic_set.rs:different_values_not_equal`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            {
                set1.pin().insert(1);
                set2.pin().insert(2);
```
---
## `test:tests/basic_set.rs:empty_sets_equal`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            assert_eq!(set1, set2);
            assert_eq!(set2, set1);
```
---
## `test:tests/basic_set.rs:from_iter_empty`
**Source:** `tests/basic_set.rs` (`test`)

```rust
use std::iter::FromIterator;

    let entries: Vec<usize> = Vec::new();
    let set: HashSet<usize> = HashSet::from_iter(entries.into_iter());

    assert_eq!(set.len(), 0)
```
---
## `test:tests/basic_set.rs:from_iter_repeated`
**Source:** `tests/basic_set.rs` (`test`)

```rust
use std::iter::FromIterator;

    let entries = vec![0, 0, 0];
    let set: HashSet<_> = HashSet::from_iter(entries.into_iter());
    let set = set.pin();
    assert_eq!(set.len(), 1);
    assert_eq!(set.iter().collect::<Vec<_>>(), vec![&0])
```
---
## `test:tests/basic_set.rs:get_empty`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        let e = set.get(&42, &guard);
        assert!(e.is_none());
```
---
## `test:tests/basic_set.rs:insert`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert_eq!(set.insert(42, &guard), true);
        assert_eq!(set.insert(42, &guard), false);
        assert_eq!(set.len(), 1);
```
---
## `test:tests/basic_set.rs:insert_and_get`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        set.insert(42, &set.guard());

        {
            let guard = set.guard();
            let e = set.get(&42, &guard).unwrap();
            assert_eq!(e, &42);
```
---
## `test:tests/basic_set.rs:insert_and_remove`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert!(set.insert(42, &guard));
        assert!(set.remove(&42, &guard));
        assert!(set.get(&42, &guard).is_none());
```
---
## `test:tests/basic_set.rs:len`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let len = if cfg!(miri) { 100
```
---
## `test:tests/basic_set.rs:new`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| drop(set()));
```
---
## `test:tests/basic_set.rs:reinsert`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert!(set.insert(42, &guard));
        assert!(!set.insert(42, &guard));
        {
            let guard = set.guard();
            let e = set.get(&42, &guard).unwrap();
            assert_eq!(e, &42);
```
---
## `test:tests/basic_set.rs:remove_empty`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert_eq!(set.remove(&42, &guard), false);
```
---
## `test:tests/basic_set.rs:retain_all_false`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        for i in 0..10 {
            set.pin().insert(i);
```
---
## `test:tests/basic_set.rs:retain_empty`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set| {
        let set = set();
        set.pin().retain(|_| false);
        assert_eq!(set.len(), 0);
```
---
## `test:tests/basic_set.rs:same_values_equal`
**Source:** `tests/basic_set.rs` (`test`)

```rust
with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            {
                set1.pin().insert(1);
                set2.pin().insert(1);
```
---
## `text_file:Cargo.toml`
**Source:** `Cargo.toml` (`text_file`)

```toml
[package]
name = "papaya"
version = "0.2.3"
authors = ["Ibraheem Ahmed <ibraheem@ibraheem.ca>"]
description = "A fast and ergonomic concurrent hash-table for read-heavy workloads."
edition = "2021"
rust-version = "1.72.0"
license = "MIT"
readme = "README.md"
repository = "https://github.com/ibraheemdev/papaya"
categories = ["algorithms", "concurrency", "data-structures"]
keywords = ["concurrent", "hashmap", "atomic", "lock-free"]
exclude = ["assets/*"]

[dependencies]
equivalent = "1"
seize = "0.5"
serde = { version = "1", optional = true }

[dev-dependencies]
rand = "0.8"
base64 = "0.22"
hdrhistogram = "7"
dashmap = "5"
criterion = "0.5"
tokio = { version = "1", features = ["fs", "rt"] }
num_cpus = "1"
serde_json = "1"

[features]
default = []
serde = ["dep:serde"]

[profile.test]
inherits = "release"
debug-assertions = true

[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = [
    'cfg(papaya_stress)',
    'cfg(papaya_asan)',
] }

[[bench]]
name = "single_thread"
harness = false

[[bench]]
name = "latency"
harness = false

```
---
## `text_file:fuzz/Cargo.toml`
**Source:** `fuzz/Cargo.toml` (`text_file`)

```toml
[package]
name = "papaya-fuzz"
version = "0.0.0"
publish = false
edition = "2021"

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
arbitrary = { features = ["derive"], version = "1.0" }

[dependencies.papaya]
path = ".."

[[bin]]
name = "std"
path = "fuzz_targets/std.rs"
test = false
doc = false
bench = false

```
