//! Cross-platform synchronization primitives
//!
//! This module provides unified types that are:
//! - Thread-safe (RwLock) on native for parallel execution
//! - Single-threaded (RefCell) on WASM for compatibility
//!
//! This enables the query executor to use parallel iteration on native
//! while maintaining WASM compatibility.

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
