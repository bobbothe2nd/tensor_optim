use alloc::{boxed::Box, vec, vec::Vec};
use lazy_simd::{
    scalar::Primitive,
    simd::{backend::NonAssociativeSimd, SimdElement},
    MAX_SIMD_SINGLE_PRECISION_LANES,
};

use crate::internal::TensorOps;

/// Moderately flexible tensors that allow for almost any data.
///
/// Compared to `RefTensor` and `ArrTensor`, the flexibility of this is nothing to be ignored.
/// In most use cases, this is the best tensor, it has a good balance of compile-time speed
/// and runtime efficiency.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynTensor<T, const LANES: usize = MAX_SIMD_SINGLE_PRECISION_LANES>
where
    T: SimdElement + Primitive,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    shape: Box<[usize]>,
    data: Box<[T]>,
}

impl<T, const LANES: usize> DynTensor<T, LANES>
where
    T: SimdElement + Primitive,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    /// Allocates a default `DynTensor` with the given shape.
    ///
    /// # Panics
    ///
    /// This constructor panics when the product of each dimension is not equal to the length of all the data.
    #[must_use]
    pub fn new(shape: &[usize]) -> Self
    where
        T: Default,
    {
        let len = shape.iter().product();
        Self::from_vec(shape, vec![T::default(); len])
    }

    /// Allocates a new `DynTensor` with the given shape and data.
    ///
    /// # Panics
    ///
    /// This constructor panics when the product of each dimension is not equal to the length of all the data.
    #[must_use]
    pub fn with_data(shape: &[usize], data: &[T]) -> Self
    where
        T: Default,
    {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape and data length mismatch"
        );
        Self {
            shape: shape.into(),
            data: data.into(),
        }
    }

    /// Allocates a new `DynTensor` from the given `Vec`.
    ///
    /// # Panics
    ///
    /// If the length of the `Vec` is not equal to the product of each dimension in the shape, this constructor panics.
    #[must_use]
    pub fn from_vec(shape: &[usize], vec: Vec<T>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), vec.len());
        Self {
            shape: shape.into(),
            data: vec.into_boxed_slice(),
        }
    }

    /// Apply a function pairwise elementwise over two `DynTensor`s, mapping to a new tensor.
    ///
    /// # Panics
    ///
    /// Both tensors, `self` and `other`, must have the same shape and data length or a panic will occur.
    pub fn zip_map<U, V, F, const L1: usize, const L2: usize>(
        &self,
        other: &DynTensor<U, L1>,
        f: F,
    ) -> DynTensor<V, L2>
    where
        U: SimdElement + Primitive,
        [U; L1]: NonAssociativeSimd<[U; L1], U, L1>,
        V: SimdElement + Primitive,
        [V; L2]: NonAssociativeSimd<[V; L2], V, L2>,
        F: Fn(&T, &U) -> V,
    {
        assert_eq!(&*self.shape, &*other.shape, "shape mismatch in zip_map");
        assert_eq!(self.data.len(), other.data.len());

        let new_data: Vec<V> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| f(a, b))
            .collect();

        DynTensor {
            shape: self.shape.clone(),
            data: new_data.into_boxed_slice(),
        }
    }

    /// Map each element of `DynTensor<T, N, D>` to `DynTensor<U, N, D>` by applying `f` elementwise.
    pub fn map<U, F, const L1: usize>(&self, f: F) -> DynTensor<U, L1>
    where
        F: Fn(&T) -> U,
        U: SimdElement + Primitive,
        [U; L1]: NonAssociativeSimd<[U; L1], U, L1>,
    {
        let new_data: Vec<U> = self.data.iter().map(f).collect();
        DynTensor {
            shape: self.shape.clone(),
            data: new_data.into_boxed_slice(),
        }
    }

    /// Returns a new `DynTensor` with axes permuted by `perm`.
    ///
    /// # Panics
    ///
    /// - if `perm` is not a valid permutation of `[0..rank]`
    /// - if `perm.len() != self.rank()`
    ///
    /// This function copies data into a new tensor with permuted axes.
    #[must_use]
    pub fn transpose_axes(&self, perm: &[usize]) -> Self {
        let rank = self.shape.len();
        assert_eq!(
            perm.len(),
            rank,
            "Permutation length must equal tensor rank"
        );

        // validate perm is a permutation of [0..rank)
        {
            let mut seen = vec![false; rank];
            for &p in perm {
                assert!(p < rank, "Invalid axis in permutation");
                assert!(!seen[p], "Duplicate axis in permutation");
                seen[p] = true;
            }
        }

        // compute new shape by permuting old shape
        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();

        // compute strides for old and new shape using your existing functions
        let old_strides = compute_strides(&self.shape);

        let n = self.data.len();
        let mut new_data = Vec::with_capacity(n);

        // temporary buffers for multi-indices
        let mut new_multi_idx = vec![0; rank];
        let mut old_multi_idx = vec![0; rank];

        for new_flat_idx in 0..n {
            unravel_index(new_flat_idx, &new_shape, &mut new_multi_idx);

            // inverse permutation: old_multi_idx[perm[i]] = new_multi_idx[i]
            for i in 0..rank {
                old_multi_idx[perm[i]] = new_multi_idx[i];
            }

            let old_flat_idx = old_multi_idx
                .iter()
                .zip(old_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();

            // clone data element from old tensor
            new_data.push(self.data[old_flat_idx]);
        }

        Self::from_vec(&new_shape, new_data)
    }

    /// Transpose the tensor through its center.
    ///
    /// Default transpose:
    ///
    /// - for 2D tensors swaps axes 0 and 1
    /// - for other ranks reverses axes order
    #[must_use]
    pub fn transpose(&self) -> Self {
        let rank = self.shape.len();
        let perm = if rank == 2 {
            vec![1, 0]
        } else {
            (0..rank).rev().collect()
        };
        self.transpose_axes(&perm)
    }
}

impl<T, const LANES: usize> TensorOps<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[T] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

use core::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};

impl<T, const LANES: usize> Index<&[usize]> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = T;
    fn index(&self, idx: &[usize]) -> &Self::Output {
        let flat = self
            .index_offset(idx)
            .expect("received invalid index into tensor");
        &self.data[flat]
    }
}

impl<T, const LANES: usize> AddAssign<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Add<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch");
        let dst = &mut self.data;
        for (a, b) in dst.iter_mut().zip(rhs.data.into_vec()) {
            *a += b;
        }
    }
}

impl<T, const LANES: usize> SubAssign<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Sub<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch");
        let dst = &mut self.data;
        for (a, b) in dst.iter_mut().zip(rhs.data.into_vec()) {
            *a -= b;
        }
    }
}

impl<T, const LANES: usize> MulAssign<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Mul<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn mul_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch");
        let dst = &mut self.data;
        for (a, b) in dst.iter_mut().zip(rhs.data.into_vec()) {
            *a *= b;
        }
    }
}

impl<T, const LANES: usize> DivAssign<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Div<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn div_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch");
        let dst = &mut self.data;
        for (a, b) in dst.iter_mut().zip(rhs.data.into_vec()) {
            *a /= b;
        }
    }
}

impl<T, const LANES: usize> AddAssign<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Add<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn add_assign(&mut self, rhs: T) {
        let dst = &mut self.data;
        for a in dst.iter_mut() {
            *a += rhs;
        }
    }
}

impl<T, const LANES: usize> SubAssign<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Sub<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn sub_assign(&mut self, rhs: T) {
        let dst = &mut self.data;
        for a in dst.iter_mut() {
            *a -= rhs;
        }
    }
}

impl<T, const LANES: usize> MulAssign<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Mul<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn mul_assign(&mut self, rhs: T) {
        let dst = &mut self.data;
        for a in dst.iter_mut() {
            *a *= rhs;
        }
    }
}

impl<T, const LANES: usize> DivAssign<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Div<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn div_assign(&mut self, rhs: T) {
        let dst = &mut self.data;
        for a in dst.iter_mut() {
            *a /= rhs;
        }
    }
}

impl<T, const LANES: usize> Add<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Add<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch in Add");
        let data: alloc::vec::Vec<T> = self
            .data
            .into_vec()
            .into_iter()
            .zip(rhs.data.into_vec())
            .map(|(a, b)| a + b)
            .collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> Sub<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Sub<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch in Sub");
        let data: alloc::vec::Vec<T> = self
            .data
            .into_vec()
            .into_iter()
            .zip(rhs.data.into_vec())
            .map(|(a, b)| a - b)
            .collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> Mul<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Mul<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch in Mul");
        let data: alloc::vec::Vec<T> = self
            .data
            .into_vec()
            .into_iter()
            .zip(rhs.data.into_vec())
            .map(|(a, b)| a * b)
            .collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> Div<Self> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Div<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "shape mismatch in Div");
        let data: alloc::vec::Vec<T> = self
            .data
            .into_vec()
            .into_iter()
            .zip(rhs.data.into_vec())
            .map(|(a, b)| a / b)
            .collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> Add<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Add<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        let data: alloc::vec::Vec<T> = self.data.iter().map(|&a| a + rhs).collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> Sub<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Sub<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        let data: alloc::vec::Vec<T> = self.data.iter().map(|&a| a - rhs).collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> Mul<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Mul<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        let data: alloc::vec::Vec<T> = self.data.iter().map(|&a| a * rhs).collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> Div<T> for DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Div<Output = T>,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        let data: alloc::vec::Vec<T> = self.data.iter().map(|&a| a / rhs).collect();
        Self::from_vec(&self.shape, data)
    }
}

impl<T, const LANES: usize> DynTensor<T, LANES>
where
    T: SimdElement + Primitive + Add<Output = T> + Mul<Output = T> + AddAssign + Default,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    /// Batched matmul over arbitrary leading batch dims:
    /// Contracts last dim of `self` with second-last dim of `rhs`.
    ///
    /// # Example
    ///
    /// - self shape: [..., M, K]
    /// - rhs shape: [..., K, N]
    /// - output shape: [..., M, N]
    ///
    /// Output buffer must be preallocated.
    ///
    /// # Panics
    ///
    /// This method panics when if any of these conditions are not met:
    ///
    /// - both tensors must have at least 2 dimensions
    /// - each tensors' inner and batch dimensions must match
    /// - the output tensor must be the correct shape
    ///
    /// Every one of those must be true.
    pub fn matmul(&self, rhs: &Self, out: &mut Self) {
        let self_rank = self.shape.len();
        let rhs_rank = rhs.shape.len();

        assert!(self_rank >= 2 && rhs_rank >= 2, "matmul requires rank >= 2");

        let m = self.shape[self_rank - 2];
        let k = self.shape[self_rank - 1];
        let n = rhs.shape[rhs_rank - 1];
        assert!(k == rhs.shape[rhs_rank - 2], "inner dimensions must match");

        let batch_shape = &self.shape[..self_rank - 2];
        assert!(
            batch_shape == &rhs.shape[..rhs_rank - 2],
            "batch dimensions must match"
        );

        let batch_count = batch_shape.iter().product::<usize>();

        // compute strides
        let self_strides = compute_strides(&self.shape);
        let rhs_strides = compute_strides(&rhs.shape);
        let out_strides = compute_strides(&out.shape);

        let out_data = &mut out.data[..];

        // zero output
        for v in out_data.iter_mut() {
            *v = T::default();
        }

        if batch_count == 0 {
            return;
        }

        // preallocate batch index buffer once
        let mut batch_multi_idx = vec![0; batch_shape.len()];

        for batch_idx in 0..batch_count {
            unravel_index(batch_idx, batch_shape, &mut batch_multi_idx);

            // compute linear batch offsets for self, rhs, and out
            let self_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&self_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            let rhs_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&rhs_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            let out_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&out_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            // inner 2D matmul per batch
            for i in 0..m {
                let self_row_offset = self_batch_offset + i * self_strides[self_rank - 2];
                let out_row_offset = out_batch_offset + i * out_strides[out.shape.len() - 2];

                for kk in 0..k {
                    let a = self.data[self_row_offset + kk];
                    let rhs_row_offset = rhs_batch_offset + kk * rhs_strides[rhs_rank - 2];

                    for j in 0..n {
                        let out_idx = out_row_offset + j;
                        out_data[out_idx] += a * rhs.data[rhs_row_offset + j];
                    }
                }
            }
        }
    }
}

impl DynTensor<f32> {
    /// SIMD-accelerated variant of [`Self::matmul`].
    ///
    /// For details, see the normal matrix multiplication documentation,
    /// do not consult these.
    ///
    /// # Panics
    ///
    /// Same preconditions as regular `matmul`
    pub fn simd_matmul(&self, rhs: &Self, out: &mut Self) {
        let self_rank = self.shape.len();
        let rhs_rank = rhs.shape.len();

        assert!(self_rank >= 2 && rhs_rank >= 2, "matmul requires rank >= 2");

        let m = self.shape[self_rank - 2];
        let k = self.shape[self_rank - 1];
        let n = rhs.shape[rhs_rank - 1];
        assert!(k == rhs.shape[rhs_rank - 2], "inner dimensions must match");

        let batch_shape = &self.shape[..self_rank - 2];
        assert!(
            batch_shape == &rhs.shape[..rhs_rank - 2],
            "batch dimensions must match"
        );

        let batch_count = batch_shape.iter().product::<usize>();

        // compute strides
        let self_strides = compute_strides(&self.shape);
        let rhs_strides = compute_strides(&rhs.shape);
        let out_strides = compute_strides(&out.shape);

        out.data.fill(0.0);

        if batch_count == 0 {
            return;
        }

        // preallocate batch index buffer once
        let mut batch_multi_idx = vec![0; batch_shape.len()];

        for batch_idx in 0..batch_count {
            unravel_index(batch_idx, batch_shape, &mut batch_multi_idx);

            // compute linear batch offsets for self, rhs, and out
            let self_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&self_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            let rhs_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&rhs_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            let out_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&out_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            // inner 2D matmul per batch
            for i in 0..m {
                let self_row_offset = self_batch_offset + i * self_strides[self_rank - 2];
                let out_row_offset = out_batch_offset + i * out_strides[self_rank - 2];
                let out_row = &mut out.data[out_row_offset..out_row_offset + n];

                for kk in 0..k {
                    let a = self.data[self_row_offset + kk];
                    let rhs_row_offset = rhs_batch_offset + kk * rhs_strides[self_rank - 2];
                    let rhs_row = &rhs.data[rhs_row_offset..(rhs_row_offset + n)];

                    lazy_simd::simd::mul_add_scalar_slice(a, rhs_row, out_row);
                }
            }
        }
    }
}

impl DynTensor<f64> {
    /// SIMD-accelerated variant of [`Self::matmul`].
    ///
    /// For details, see the normal matrix multiplication documentation,
    /// do not consult these.
    ///
    /// # Panics
    ///
    /// Same preconditions as regular `matmul`
    pub fn simd_matmul(&self, rhs: &Self, out: &mut Self) {
        let self_rank = self.shape.len();
        let rhs_rank = rhs.shape.len();

        assert!(self_rank >= 2 && rhs_rank >= 2, "matmul requires rank >= 2");

        let m = self.shape[self_rank - 2];
        let k = self.shape[self_rank - 1];
        let n = rhs.shape[rhs_rank - 1];
        assert!(k == rhs.shape[rhs_rank - 2], "inner dimensions must match");

        let batch_shape = &self.shape[..self_rank - 2];
        assert!(
            batch_shape == &rhs.shape[..rhs_rank - 2],
            "batch dimensions must match"
        );

        let batch_count = batch_shape.iter().product::<usize>();

        // compute strides
        let self_strides = compute_strides(&self.shape);
        let rhs_strides = compute_strides(&rhs.shape);
        let out_strides = compute_strides(&out.shape);

        out.data.fill(0.0);

        if batch_count == 0 {
            return;
        }

        // preallocate batch index buffer once
        let mut batch_multi_idx = vec![0; batch_shape.len()];

        for batch_idx in 0..batch_count {
            unravel_index(batch_idx, batch_shape, &mut batch_multi_idx);

            // compute linear batch offsets for self, rhs, and out
            let self_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&self_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            let rhs_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&rhs_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            let out_batch_offset: usize = batch_multi_idx
                .iter()
                .zip(&out_strides[..batch_shape.len()])
                .map(|(&i, &s)| i * s)
                .sum();

            // inner 2D matmul per batch
            for i in 0..m {
                let self_row_offset = self_batch_offset + i * self_strides[self_rank - 2];
                let out_row_offset = out_batch_offset + i * out_strides[self_rank - 2];
                let out_row = &mut out.data[out_row_offset..out_row_offset + n];

                for kk in 0..k {
                    let a = self.data[self_row_offset + kk];
                    let rhs_row_offset = rhs_batch_offset + kk * rhs_strides[self_rank - 2];
                    let rhs_row = &rhs.data[rhs_row_offset..(rhs_row_offset + n)];

                    lazy_simd::simd::mul_add_scalar_slice_double(a, rhs_row, out_row);
                }
            }
        }
    }
}

// compute strides for dynamic shape
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

// unravel linear index into multidim index vector
fn unravel_index(mut idx: usize, shape: &[usize], out: &mut [usize]) {
    for i in (0..shape.len()).rev() {
        out[i] = idx % shape[i];
        idx /= shape[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: &[f32], shape: &[usize]) -> DynTensor<f32> {
        DynTensor::with_data(shape, data)
    }

    #[test]
    fn indexing() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t[&[0, 0]], 1.0);
        assert_eq!(t[&[1, 2]], 6.0);
        assert!(t.index_offset(&[2, 0]).is_none());
    }

    #[test]
    fn add_assign_tensor() {
        let mut a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        a += b;
        assert_eq!(a.data.as_ref(), [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_assign_shape_mismatch() {
        let mut a = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        let b = make_tensor(&[4.0, 5.0], &[2]);
        a += b;
    }

    #[test]
    fn sub_assign_tensor() {
        let mut a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let b = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        a -= b;
        assert_eq!(a.data.as_ref(), [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn mul_assign_tensor() {
        let mut a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        a *= b;
        assert_eq!(a.data.as_ref(), [10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn div_assign_tensor() {
        let mut a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let b = make_tensor(&[2.0, 4.0, 5.0, 10.0], &[2, 2]);
        a /= b;
        assert_eq!(a.data.as_ref(), [5.0, 5.0, 6.0, 4.0]);
    }

    #[test]
    fn add_assign_scalar() {
        let mut a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        a += 5.0;
        assert_eq!(a.data.as_ref(), [6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn sub_assign_scalar() {
        let mut a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        a -= 10.0;
        assert_eq!(a.data.as_ref(), [0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn mul_assign_scalar() {
        let mut a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        a *= 3.0;
        assert_eq!(a.data.as_ref(), [3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn div_assign_scalar() {
        let mut a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        a /= 10.0;
        assert_eq!(a.data.as_ref(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn add_tensor_consuming() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a + b;
        assert_eq!(c.data.as_ref(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn sub_tensor_consuming() {
        let a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let b = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = a - b;
        assert_eq!(c.data.as_ref(), &[9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn mul_tensor_consuming() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let c = a * b;
        assert_eq!(c.data.as_ref(), &[10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn div_tensor_consuming() {
        let a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let b = make_tensor(&[2.0, 4.0, 5.0, 10.0], &[2, 2]);
        let c = a / b;
        assert_eq!(c.data.as_ref(), &[5.0, 5.0, 6.0, 4.0]);
    }

    #[test]
    fn add_scalar_consuming() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = a + 10.0;
        assert_eq!(c.data.as_ref(), &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn sub_scalar_consuming() {
        let a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let c = a - 5.0;
        assert_eq!(c.data.as_ref(), &[5.0, 15.0, 25.0, 35.0]);
    }

    #[test]
    fn mul_scalar_consuming() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = a * 2.0;
        assert_eq!(c.data.as_ref(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn div_scalar_consuming() {
        let a = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let c = a / 10.0;
        assert_eq!(c.data.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn batched_matmul_simple() {
        // Shape: [2, 2, 3] (2 batches, 2 rows, 3 cols)
        let a_data = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let a: DynTensor<f64> = DynTensor::with_data(&[2, 2, 3], &a_data);

        // Shape: [2, 3, 2] (2 batches, 3 rows, 2 cols)
        let b_data = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let b: DynTensor<f64> = DynTensor::with_data(&[2, 3, 2], &b_data);

        let mut out: DynTensor<f64> = DynTensor::new(&[2, 2, 2]);

        a.matmul(&b, &mut out);

        let expected = [22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0];

        assert_eq!(out.data.as_ref(), expected);
    }
}
