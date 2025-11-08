//! This library offers basic tensor types:
//!
//! - `RefTensor`: Statically sized slices with a mutable reference for data.
//!     - Ownership: The tensor does not own it's data, it has references for the lifetime `'a` to each slice.
//!     - Allocation: This tensor has no dynamic allocation, it doesn't have it's own memory, just pointers.
//! - `ArrTensor`: Statically sized arrays build the foundation for shape and data memory.
//!     - Ownership: The entire tensor is owned by the `struct`.
//!     - Allocation: This tensor has no dynamic allocation, it lives on the stack.
//! - `DynTensor`: A dynamically allocated tensor with far more flexibility than the others.
//!     - Ownership: The entire tensor is owned by the `struct`.
//!     - Allocation: This tensor dynamically allocates everything, shape is boxed and data is wrapped in `Arc`.
//!
//! Note: The crate is fully documented, `no-std` compatible, and well tested.
//! It doesn't even need `alloc` unless the `alloc` feature (off by default) is enabled.

#![forbid(missing_docs)]
#![forbid(unsafe_code)]
#![forbid(clippy::nursery)]
#![forbid(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::many_single_char_names)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

mod internal;

pub use internal::array::ArrTensor;
pub use internal::views::RefTensor;
pub use internal::ConstTensorOps;
pub use internal::TensorOps;
pub use internal::MAX_STATIC_RANK;

#[cfg(feature = "alloc")]
pub use internal::dynamic::DynTensor;
