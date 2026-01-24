//! Cross-platform synchronization primitives
//!
//! This module provides unified types that are:
//! - Thread-safe (RwLock) on native for parallel execution
//! - Single-threaded (RefCell) on WASM for compatibility
//!
//! This enables the query executor to use parallel iteration on native
//! while maintaining WASM compatibility.
//!
//! # Error-Safe Lock Access
//!
//! The `LockExt` trait provides methods that return `LatticeResult` instead
//! of panicking on lock poisoning, enabling enterprise-grade error handling.
//!
//! # Float Comparison
//!
//! The `cmp_f32` function provides NaN-safe float comparison for sorting.

// =============================================================================
// Native implementation (parking_lot RwLock)
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use parking_lot::RwLock;
    use std::ops::Deref;

    /// A cell that provides interior mutability with thread-safe access on native.
    /// Uses parking_lot::RwLock which is faster than std::sync::RwLock.
    pub struct SyncCell<T>(RwLock<T>);

    impl<T> SyncCell<T> {
        /// Create a new SyncCell with the given value.
        #[inline]
        pub fn new(value: T) -> Self {
            Self(RwLock::new(value))
        }

        /// Borrow the contained value immutably.
        /// On native, this acquires a read lock (concurrent reads allowed).
        #[inline]
        pub fn borrow(&self) -> impl Deref<Target = T> + '_ {
            self.0.read()
        }

        /// Borrow the contained value mutably.
        /// On native, this acquires a write lock (exclusive access).
        #[inline]
        pub fn borrow_mut(&self) -> impl std::ops::DerefMut<Target = T> + '_ {
            self.0.write()
        }
    }

    // SAFETY: SyncCell is safe to share between threads because it uses RwLock internally
    unsafe impl<T: Send> Send for SyncCell<T> {}
    unsafe impl<T: Send + Sync> Sync for SyncCell<T> {}
}

// =============================================================================
// WASM implementation (RefCell - single-threaded)
// =============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::RefCell;
    use std::ops::Deref;

    /// A cell that provides interior mutability.
    /// On WASM, this is just a RefCell (no threading support needed).
    pub struct SyncCell<T>(RefCell<T>);

    impl<T> SyncCell<T> {
        /// Create a new SyncCell with the given value.
        #[inline]
        pub fn new(value: T) -> Self {
            Self(RefCell::new(value))
        }

        /// Borrow the contained value immutably.
        #[inline]
        pub fn borrow(&self) -> impl Deref<Target = T> + '_ {
            self.0.borrow()
        }

        /// Borrow the contained value mutably.
        #[inline]
        pub fn borrow_mut(&self) -> impl std::ops::DerefMut<Target = T> + '_ {
            self.0.borrow_mut()
        }
    }
}

// =============================================================================
// Public re-exports
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

// =============================================================================
// Error-Safe Lock Extension Trait
// =============================================================================

/// Error codes for internal synchronization errors
pub mod error_codes {
    /// RwLock poisoned during read operation
    pub const RWLOCK_READ_POISON: u32 = 50001;
    /// RwLock poisoned during write operation
    pub const RWLOCK_WRITE_POISON: u32 = 50002;
    /// Mutex poisoned
    pub const MUTEX_POISON: u32 = 50003;
}

#[cfg(not(target_arch = "wasm32"))]
mod lock_ext {
    use crate::{LatticeError, LatticeResult};
    use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

    /// Extension trait for std::sync::RwLock that returns `LatticeResult`
    /// instead of panicking on lock poisoning.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::RwLock;
    /// use lattice_core::sync::LockExt;
    ///
    /// let lock = RwLock::new(42);
    /// let guard = lock.read_safe()?;  // Returns LatticeResult
    /// ```
    pub trait LockExt<T> {
        /// Acquire a read lock, returning an error instead of panicking on poison
        fn read_safe(&self) -> LatticeResult<RwLockReadGuard<'_, T>>;

        /// Acquire a write lock, returning an error instead of panicking on poison
        fn write_safe(&self) -> LatticeResult<RwLockWriteGuard<'_, T>>;
    }

    impl<T> LockExt<T> for RwLock<T> {
        fn read_safe(&self) -> LatticeResult<RwLockReadGuard<'_, T>> {
            self.read().map_err(|e| LatticeError::Internal {
                code: super::error_codes::RWLOCK_READ_POISON,
                message: format!("RwLock poisoned during read: {}", e),
            })
        }

        fn write_safe(&self) -> LatticeResult<RwLockWriteGuard<'_, T>> {
            self.write().map_err(|e| LatticeError::Internal {
                code: super::error_codes::RWLOCK_WRITE_POISON,
                message: format!("RwLock poisoned during write: {}", e),
            })
        }
    }

    /// Extension trait for std::sync::Mutex that returns `LatticeResult`
    /// instead of panicking on mutex poisoning.
    pub trait MutexExt<T> {
        /// Acquire the mutex, returning an error instead of panicking on poison
        fn lock_safe(&self) -> LatticeResult<MutexGuard<'_, T>>;
    }

    impl<T> MutexExt<T> for Mutex<T> {
        fn lock_safe(&self) -> LatticeResult<MutexGuard<'_, T>> {
            self.lock().map_err(|e| LatticeError::Internal {
                code: super::error_codes::MUTEX_POISON,
                message: format!("Mutex poisoned: {}", e),
            })
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use lock_ext::{LockExt, MutexExt};

// =============================================================================
// Float Comparison Utilities
// =============================================================================

/// Compare f32 values with proper NaN handling
///
/// NaN values are sorted to the end (greater than all other values).
/// This is semantically correct for similarity search where NaN indicates
/// invalid/corrupted data that should be deprioritized.
///
/// # Ordering Rules
///
/// - Normal values: standard floating-point comparison
/// - NaN vs Normal: NaN is greater (sorts to end)
/// - NaN vs NaN: Equal
///
/// # Performance
///
/// This function is `#[inline]` and has zero allocation overhead.
#[inline]
pub fn cmp_f32(a: f32, b: f32) -> std::cmp::Ordering {
    // Fast path: most comparisons succeed
    if let Some(ord) = a.partial_cmp(&b) {
        return ord;
    }

    // Slow path: at least one NaN - sort NaN to end
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => std::cmp::Ordering::Equal, // unreachable
    }
}

/// Compare f32 values in reverse order (for max-heap behavior)
#[inline]
pub fn cmp_f32_reverse(a: f32, b: f32) -> std::cmp::Ordering {
    cmp_f32(b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_cmp_f32_normal_values() {
        assert_eq!(cmp_f32(1.0, 2.0), Ordering::Less);
        assert_eq!(cmp_f32(2.0, 1.0), Ordering::Greater);
        assert_eq!(cmp_f32(1.0, 1.0), Ordering::Equal);
        assert_eq!(cmp_f32(-1.0, 1.0), Ordering::Less);
    }

    #[test]
    fn test_cmp_f32_nan_handling() {
        let nan = f32::NAN;
        assert_eq!(cmp_f32(nan, 1.0), Ordering::Greater);
        assert_eq!(cmp_f32(1.0, nan), Ordering::Less);
        assert_eq!(cmp_f32(nan, nan), Ordering::Equal);
    }

    #[test]
    fn test_cmp_f32_sort_stability() {
        let mut values = [3.0, f32::NAN, 1.0, f32::NAN, 2.0];
        values.sort_by(|a, b| cmp_f32(*a, *b));
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod lock_tests {
        use super::super::LockExt;
        use std::sync::RwLock;

        #[test]
        fn test_lock_ext_read() {
            let lock = RwLock::new(42);
            let guard = lock.read_safe().unwrap();
            assert_eq!(*guard, 42);
        }

        #[test]
        fn test_lock_ext_write() {
            let lock = RwLock::new(42);
            {
                let mut guard = lock.write_safe().unwrap();
                *guard = 100;
            }
            let guard = lock.read_safe().unwrap();
            assert_eq!(*guard, 100);
        }
    }
}
