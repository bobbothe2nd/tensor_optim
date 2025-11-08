# `tensor_optim`

A no-std compatible, zero-unsafe public API Rust library providing foundational tensor abstractions optimized for embedded and performance-critical machine learning workloads.

## Features

- No unsafe code in the entire library — safety guaranteed by design.
- Supports `no_std` by default; dynamic allocation disabled by default.
- Optional `alloc` feature enables dynamically sized, heap-allocated tensors (`DynTensor`).
- Implements elementwise arithmetic traits and indexing for ergonomic tensor operations.
- Provides matrix multiplication and transpose primitives optimized for minimal overhead.
- Extensive test coverage with strict clippy lints and no missing docs.

## Overview

`tensor_optim` offers three primary tensor types representing different ownership and allocation models:

| Type         | Ownership                  | Allocation                 | Use Case                                              |
|--------------|----------------------------|----------------------------|-------------------------------------------------------|
| `RefTensor`* | Borrowed mutable slices    | None (zero-copy views)     | Temporary tensor views, zero-allocation borrowing     |
| `ArrTensor`  | Fully owned, fixed size    | Stack                      | Fixed-shape tensors in embedded or realtime systems   |
| `DynTensor`  | Fully owned, dynamic size  | Heap (behind `alloc` flag) | Flexible tensor sizes with shared ownership via `Arc` |

[*] `RefTensor` is a special tensor type that doesn't own anything and lacks capability like matrix multiplication and transposition.

## Safety and Performance

- The **public API is 100% safe Rust**; no `unsafe` code is exposed or required by users.
- The design delivers near-zero-cost abstractions without compromising soundness or correctness.

## Explanation

- Use `RefTensor` to create zero-copy views into existing data without allocation or ownership transfer.
- Use `ArrTensor` for fixed-size tensors with deterministic, stack-allocated memory layout.
- Enable the `alloc` feature and use `DynTensor` for dynamically sized tensors requiring heap allocation and shared ownership.

All tensor types support (with the exception of `RefTensor`):

- Elementwise arithmetic via standard Rust operator traits (`Add`, `Sub`, `Mul`, `Div` and assignment variants).
- Indexing with `Index` and `IndexMut` for direct element access.
- Matrix multiplication and transpose operations optimized for minimal overhead.

... and every tensor is row-major.

## Feature Flags

- `alloc` (disabled by default): Enables `DynTensor` and dynamic heap allocation. Requires allocator support.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tensor_optim = "0.2.1"
# or enable dynamic tensors:
# tensor_optim = { version = "0.2.1", features = ["alloc"] }
```

or just run this command: `cargo add tensor_optim`.

## `#![no_std]` Support

- `tensor_optim` compiles without the Rust standard library, always.
- To use dynamic tensors (`DynTensor`), enable the `alloc` feature.

## Testing and Quality

- Extensively tested for correctness, edge cases, and safety.
- Zero unsafe in public API, internal unsafe is forbidden.
- Strict `clippy` lint enforcement and fulll documentation coverage.

## Intended Use Cases

- Embeded ML inference on microcontrollers and real-time systems.
- Kernel or OS-level tensor operations requiring zero runtime overhead.
- Resource-constrained environments where deterministic memory usage is important.
- Foundation for building higher-level Rust ML frameworks or inference engines.

## Licensing & Contribution

Licensed under an MIT license.

`tensor_optim` is a minimal, safe, and high-performance Rust tensor library that empowers embedded and performance-critical ML applications without sacrificing safety or compatibility. Its carefully designed ownership models and optimized core kernels provide the solid foundation needed for next-generation Rust ML frameworks.
