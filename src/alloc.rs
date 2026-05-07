use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Debug-only allocator wrapper that tracks allocation count and bytes.
#[cfg(debug_assertions)]
pub struct TrackingAllocator;

#[cfg(debug_assertions)]
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Reset allocation counters to zero.
#[cfg(debug_assertions)]
pub fn reset_alloc_stats() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

/// Get current allocation stats as `(count, total_bytes)`.
#[cfg(debug_assertions)]
pub fn get_alloc_stats() -> (usize, usize) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_clears_stats() {
        reset_alloc_stats();
        let (count, bytes) = get_alloc_stats();
        assert_eq!(count, 0, "count should be zero after reset");
        assert_eq!(bytes, 0, "bytes should be zero after reset");
    }

    #[test]
    fn test_alloc_increments_count() {
        reset_alloc_stats();
        let _v: Vec<u8> = vec![0u8; 1024];
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "at least one allocation should have occurred");
        assert!(bytes >= 1024, "bytes should be at least 1024, got {bytes}");
    }

    #[test]
    fn test_multiple_allocs_accumulate() {
        reset_alloc_stats();
        let _v1: Vec<u8> = vec![0u8; 64];
        let _v2: Vec<u8> = vec![0u8; 128];
        let (count, bytes) = get_alloc_stats();
        assert!(count >= 2, "at least two allocations, got {count}");
        assert!(bytes >= 192, "bytes should be at least 192, got {bytes}");
    }

    #[test]
    fn test_string_allocation() {
        reset_alloc_stats();
        let _s = String::from("hello world test allocation");
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "string allocation should increment count");
        assert!(bytes > 0, "string allocation should increment bytes");
    }

    #[test]
    fn test_box_allocation() {
        reset_alloc_stats();
        let _b = Box::new(42u64);
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "box allocation should increment count");
        assert!(
            bytes >= 8,
            "box allocation should account for u64, got {bytes}"
        );
    }
}
