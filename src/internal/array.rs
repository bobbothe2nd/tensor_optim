use crate::internal::{ConstTensorOps, TensorOps};
use core::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};

/// A tensor made up of statically sized arrays.
///
/// Often the best choice for embedded tensor operations because it doesn't use any OS-dependent features like heap allocators.
/// If memory efficiency is the largest concern, the lack of dynamic heap allocation is a huge positive of `ArrTensor`.
///
/// However, when flexibility is put before memory efficiency, this becomes obsolete; use `DynTensor` instead.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArrTensor<T, const N: usize, const D: usize> {
    shape: [usize; D],
    data: [T; N],
}

impl<T, const N: usize, const D: usize> ArrTensor<T, N, D> {
    /// Instantiates a new Tensor which owns its data without allocating it.
    ///
    /// # Panics
    ///
    /// This constructor panics when the product of each dimension is not equal to the length of all the data.
    #[must_use]
    pub fn new(shape: [usize; D]) -> Self
    where
        T: Default + Copy,
    {
        assert_eq!(
            shape.iter().product::<usize>(),
            N,
            "shape and data length mismatch"
        );
        Self {
            shape,
            data: [T::default(); N],
        }
    }

    /// Instantiates a new Tensor with data.
    ///
    /// # Panics
    ///
    /// This constructor panics when the product of each dimension is not equal to the length of all the data.
    ///
    /// Example:
    ///
    /// ```rust
    /// use tensor_optim::ArrTensor;
    ///
    /// const SHAPE: [usize; 2] = [2, 3];
    ///
    /// let data = [0f64; 6];
    /// let mut tensor = ArrTensor::with_data(SHAPE, data);
    ///
    /// tensor += 42.0;
    /// println!("First element: {}", tensor[&[0, 0]]);
    /// ```
    #[must_use]
    pub fn with_data(shape: [usize; D], data: [T; N]) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            N,
            "shape and data length mismatch"
        );
        Self { shape, data }
    }
}

impl<T, const N: usize, const D: usize> ArrTensor<T, N, D>
where
    T: Copy,
{
    /// Map each element of `ArrTensor<T, N, D>` to `ArrTensor<U, N, D>` by applying `f` elementwise.
    pub fn map<U, F>(&self, mut f: F) -> ArrTensor<U, N, D>
    where
        F: FnMut(T) -> U,
        U: Copy,
    {
        let new_data = core::array::from_fn(|i| f(self.data[i]));
        ArrTensor {
            shape: self.shape,
            data: new_data,
        }
    }

    /// Apply a function pairwise elementwise over two `ArrTensor`s, mapping to a new tensor.
    ///
    /// # Panics
    ///
    /// Both tensors, `self` and `other`, must have the same shape or a panic will occur.
    pub fn zip_map<U, V, F>(&self, other: &ArrTensor<U, N, D>, mut f: F) -> ArrTensor<V, N, D>
    where
        U: Copy,
        F: FnMut(T, U) -> V,
    {
        assert_eq!(self.shape, other.shape, "shape mismatch in zip_map");

        let new_data = core::array::from_fn(|i| f(self.data[i], other.data[i]));

        ArrTensor {
            shape: self.shape,
            data: new_data,
        }
    }
}

impl<T, const N: usize, const D: usize> TensorOps<T> for ArrTensor<T, N, D> {
    fn data(&self) -> &[T] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T, const N: usize, const D: usize> ConstTensorOps<T, N, D> for ArrTensor<T, N, D> {
    fn data_array(&self) -> &[T; N] {
        &self.data
    }

    fn data_mut_array(&mut self) -> &mut [T; N] {
        &mut self.data
    }

    fn shape_array(&self) -> &[usize; D] {
        &self.shape
    }
}

impl<T, const N: usize, const D: usize> Index<&[usize]> for ArrTensor<T, N, D> {
    type Output = T;
    fn index(&self, idx: &[usize]) -> &Self::Output {
        let flat = self
            .index_offset(idx)
            .expect("recieved invalid index into tensor");
        &self.data[flat]
    }
}

impl<T, const N: usize, const D: usize> AddAssign<&Self> for ArrTensor<T, N, D>
where
    T: Copy + Add<Output = T>,
{
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a + *b;
        }
    }
}

impl<T, const N: usize, const D: usize> SubAssign<&Self> for ArrTensor<T, N, D>
where
    T: Copy + Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a - *b;
        }
    }
}

impl<T, const N: usize, const D: usize> MulAssign<&Self> for ArrTensor<T, N, D>
where
    T: Copy + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a * *b;
        }
    }
}

impl<T, const N: usize, const D: usize> DivAssign<&Self> for ArrTensor<T, N, D>
where
    T: Copy + Div<Output = T>,
{
    fn div_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a / *b;
        }
    }
}

impl<T, const N: usize, const D: usize> AddAssign<T> for ArrTensor<T, N, D>
where
    T: Copy + Add<Output = T>,
{
    fn add_assign(&mut self, rhs: T) {
        for a in &mut self.data {
            *a = *a + rhs;
        }
    }
}

impl<T, const N: usize, const D: usize> SubAssign<T> for ArrTensor<T, N, D>
where
    T: Copy + Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: T) {
        for a in &mut self.data {
            *a = *a - rhs;
        }
    }
}

impl<T, const N: usize, const D: usize> MulAssign<T> for ArrTensor<T, N, D>
where
    T: Copy + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: T) {
        for a in &mut self.data {
            *a = *a * rhs;
        }
    }
}

impl<T, const N: usize, const D: usize> DivAssign<T> for ArrTensor<T, N, D>
where
    T: Copy + Div<Output = T>,
{
    fn div_assign(&mut self, rhs: T) {
        for a in &mut self.data {
            *a = *a / rhs;
        }
    }
}

impl<T, const N: usize, const D: usize> Add for ArrTensor<T, N, D>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Add");
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] + rhs.data[i];
        }
        Self::with_data(self.shape, data)
    }
}

impl<T, const N: usize, const D: usize> Sub for ArrTensor<T, N, D>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Sub");
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] - rhs.data[i];
        }
        Self::with_data(self.shape, data)
    }
}

impl<T, const N: usize, const D: usize> Mul for ArrTensor<T, N, D>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Mul");
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] * rhs.data[i];
        }
        Self::with_data(self.shape, data)
    }
}

impl<T, const N: usize, const D: usize> Div for ArrTensor<T, N, D>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Div");
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] / rhs.data[i];
        }
        Self::with_data(self.shape, data)
    }
}

impl<T, const N: usize, const D: usize> Add<T> for ArrTensor<T, N, D>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] + rhs;
        }
        Self::with_data(self.shape, data)
    }
}

impl<T, const N: usize, const D: usize> Sub<T> for ArrTensor<T, N, D>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] - rhs;
        }
        Self::with_data(self.shape, data)
    }
}

impl<T, const N: usize, const D: usize> Mul<T> for ArrTensor<T, N, D>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] * rhs;
        }
        Self::with_data(self.shape, data)
    }
}

impl<T, const N: usize, const D: usize> Div<T> for ArrTensor<T, N, D>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        let mut data = [self.data[0]; N];
        for (i, item) in data.iter_mut().enumerate().take(N) {
            *item = self.data[i] / rhs;
        }
        Self::with_data(self.shape, data)
    }
}

use super::MAX_STATIC_RANK;

fn compute_strides_fixed(shape: &[usize], out: &mut [usize; MAX_STATIC_RANK], rank: usize) {
    let mut stride = 1;
    for i in (0..rank).rev() {
        out[i] = stride;
        stride *= shape[i];
    }
}

fn unravel_index_fixed(
    mut idx: usize,
    shape: &[usize],
    out: &mut [usize; MAX_STATIC_RANK],
    rank: usize,
) {
    for i in (0..rank).rev() {
        out[i] = idx % shape[i];
        idx /= shape[i];
    }
}

impl<T, const N: usize, const D: usize> ArrTensor<T, N, D>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + AddAssign + Default,
{
    /// Batched matmul over arbitrary leading batch dims:
    /// Contracts last dim of `self` with second-last dim of `rhs`.
    ///
    /// Matrix multiplication cannot be performed on `ArrTensor`s when the dimensions exceed `MAX_STATIC_RANK`.
    /// To bypass this limit, `DynTensor` can be used which is allocated on the heap.
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
    pub fn matmul<const M: usize, const O: usize>(
        &self,
        rhs: &ArrTensor<T, M, D>,
        out: &mut ArrTensor<T, O, D>,
    ) {
        assert!(D >= 2 && D <= MAX_STATIC_RANK, "rank must be >=2 and <= MAX_STATIC_RANK");

        let m = self.shape[D - 2];
        let k = self.shape[D - 1];
        let n = rhs.shape[D - 1];
        assert!(k == rhs.shape[D - 2], "inner dimensions must match");
        assert!(self.shape[..D - 2] == rhs.shape[..D - 2], "batch dimensions must match");
        assert!(self.shape[..D - 2] == out.shape[..D - 2], "batch dimensions must match output");

        // compute strides once
        let mut self_strides = [0usize; MAX_STATIC_RANK];
        let mut rhs_strides = [0usize; MAX_STATIC_RANK];
        let mut out_strides = [0usize; MAX_STATIC_RANK];
        compute_strides_fixed(&self.shape, &mut self_strides, D);
        compute_strides_fixed(&rhs.shape, &mut rhs_strides, D);
        compute_strides_fixed(&out.shape, &mut out_strides, D);

        // zero output
        for v in &mut out.data {
            *v = T::default();
        }

        let batch_count = self.shape[..D - 2].iter().product::<usize>();
        if batch_count == 0 { return; }

        let mut batch_multi_idx = [0usize; MAX_STATIC_RANK];

        for batch_idx in 0..batch_count {
            // unravel batch index into multi-index
            unravel_index_fixed(batch_idx, &self.shape[..D - 2], &mut batch_multi_idx, D - 2);

            // compute linear batch offsets
            let self_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter().zip(&self_strides[..D - 2]).map(|(&i, &s)| i * s).sum();
            let rhs_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter().zip(&rhs_strides[..D - 2]).map(|(&i, &s)| i * s).sum();
            let out_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter().zip(&out_strides[..D - 2]).map(|(&i, &s)| i * s).sum();

            // linear inner matmul over last two dimensions
            for i in 0..m {
                let self_row_offset = self_batch_offset + i * self_strides[D - 2];
                let out_row_offset = out_batch_offset + i * out_strides[D - 2];

                for kk in 0..k {
                    let a = self.data[self_row_offset + kk];
                    let rhs_row_offset = rhs_batch_offset + kk * rhs_strides[D - 2];

                    for j in 0..n {
                        let b = rhs.data[rhs_row_offset + j];
                        out.data[out_row_offset + j] += a * b;
                    }
                }
            }
        }
    }
}

impl<T, const N: usize, const D: usize> ArrTensor<T, N, D>
where
    T: Copy,
{
    /// Transposes the tensor using a default axis permutation:
    /// - For 2D tensors, swaps the two axes.
    /// - For higher-rank tensors, reverses the axes.
    ///
    /// # Panics
    ///
    /// Panics if `D` exceeds `MAX_STATIC_RANK`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensor_optim::{ArrTensor, TensorOps};
    ///
    /// let tensor = ArrTensor::<i32, 6, 2>::with_data([2, 3], [1, 2, 3, 4, 5, 6]);
    /// let transposed = tensor.transpose();
    /// assert_eq!(transposed.shape(), [3, 2]);
    /// assert_eq!(transposed.data(), [1, 4, 2, 5, 3, 6]);
    /// ```
    #[must_use]
    pub fn transpose(&self) -> Self {
        let perm = {
            // Reverse axes for ranks > 2
            let mut rev = [0usize; D];
            let mut i = 0;
            while i < D {
                rev[i] = D - 1 - i;
                i += 1;
            }
            rev
        };

        self.transpose_axes(perm)
    }

    /// Returns a new `ArrTensor` with axes permuted according to `perm`.
    ///
    /// # Panics
    ///
    /// - If `perm` is not a permutation of `[0..D]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensor_optim::{ArrTensor, TensorOps};
    ///
    /// let tensor = ArrTensor::<i32, 6, 2>::with_data([2, 3], [1, 2, 3, 4, 5, 6]);
    /// let transposed = tensor.transpose_axes([1, 0]);
    /// assert_eq!(transposed.shape(), [3, 2]);
    /// assert_eq!(transposed.data(), [1, 4, 2, 5, 3, 6]);
    /// ```
    #[must_use]
    pub fn transpose_axes(&self, perm: [usize; D]) -> Self {
        // validate perm is a valid permutation of 0..D
        {
            let mut check = [false; D];
            for &p in &perm {
                assert!(p < D, "transpose_axes: invalid axis in permutation");
                assert!(!check[p], "transpose_axes: duplicate axis in permutation");
                check[p] = true;
            }
        }

        // compute new shape by permuting old shape
        let mut new_shape = [0usize; D];
        for i in 0..D {
            new_shape[i] = self.shape[perm[i]];
        }

        // compute old strides and new strides
        let mut old_strides = [0usize; MAX_STATIC_RANK];
        let mut new_strides = [0usize; MAX_STATIC_RANK];
        compute_strides_fixed(&self.shape, &mut old_strides, D);
        compute_strides_fixed(&new_shape, &mut new_strides, D);

        // allocate new data array
        let mut new_data = [self.data[0]; N];

        // for every flat index in new_data, find corresponding index in self.data
        for (new_flat_index, item) in new_data.iter_mut().enumerate().take(N) {
            // unravel new_flat_index to multi-dim index in permuted axes
            let mut new_multi_index = [0usize; MAX_STATIC_RANK];
            unravel_index_fixed(new_flat_index, &new_shape, &mut new_multi_index, D);

            // invert permutation: find old_multi_index by mapping
            // old_multi_index[perm[i]] = new_multi_index[i]
            let mut old_multi_index = [0usize; D];
            for i in 0..D {
                old_multi_index[perm[i]] = new_multi_index[i];
            }

            // flatten old_multi_index to get original flat index
            let old_flat_index = old_multi_index
                .iter()
                .zip(old_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();

            *item = self.data[old_flat_index];
        }

        Self {
            shape: new_shape,
            data: new_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{ArrTensor, TensorOps};

    // helper constructor
    fn tensor<T, const N: usize, const D: usize>(
        shape: [usize; D],
        buf: [T; N],
    ) -> ArrTensor<T, N, D> {
        ArrTensor::with_data(shape, buf)
    }

    #[test]
    fn add_assign_tensor() {
        let mut a = tensor([3], [1, 2, 3]);
        let b = tensor([3], [4, 5, 6]);
        a += &b;
        assert_eq!(a.data, [5, 7, 9]);
    }

    #[test]
    fn sub_assign_tensor() {
        let mut a = tensor([3], [5, 7, 9]);
        let b = tensor([3], [4, 5, 6]);
        a -= &b;
        assert_eq!(a.data, [1, 2, 3]);
    }

    #[test]
    fn mul_assign_tensor() {
        let mut a = tensor([3], [1, 2, 3]);
        let b = tensor([3], [4, 5, 6]);
        a *= &b;
        assert_eq!(a.data, [4, 10, 18]);
    }

    #[test]
    fn div_assign_tensor() {
        let mut a = tensor([3], [4, 10, 18]);
        let b = tensor([3], [4, 5, 6]);
        a /= &b;
        assert_eq!(a.data, [1, 2, 3]);
    }

    #[test]
    fn add_assign_scalar() {
        let mut a = tensor([3], [1, 2, 3]);
        a += 1;
        assert_eq!(a.data, [2, 3, 4]);
    }

    #[test]
    fn sub_assign_scalar() {
        let mut a = tensor([3], [5, 6, 7]);
        a -= 2;
        assert_eq!(a.data, [3, 4, 5]);
    }

    #[test]
    fn mul_assign_scalar() {
        let mut a = tensor([3], [1, 2, 3]);
        a *= 3;
        assert_eq!(a.data, [3, 6, 9]);
    }

    #[test]
    fn div_assign_scalar() {
        let mut a = tensor([3], [4, 6, 8]);
        a /= 2;
        assert_eq!(a.data, [2, 3, 4]);
    }

    #[test]
    fn add_tensor() {
        let a = tensor([3], [1, 2, 3]);
        let b = tensor([3], [4, 5, 6]);
        let c = a + b;
        assert_eq!(c.data, [5, 7, 9]);
    }

    #[test]
    fn sub_tensor() {
        let a = tensor([3], [5, 7, 9]);
        let b = tensor([3], [4, 5, 6]);
        let c = a - b;
        assert_eq!(c.data, [1, 2, 3]);
    }

    #[test]
    fn mul_tensor() {
        let a = tensor([3], [1, 2, 3]);
        let b = tensor([3], [4, 5, 6]);
        let c = a * b;
        assert_eq!(c.data, [4, 10, 18]);
    }

    #[test]
    fn div_tensor() {
        let a = tensor([3], [4, 10, 18]);
        let b = tensor([3], [4, 5, 6]);
        let c = a / b;
        assert_eq!(c.data, [1, 2, 3]);
    }

    #[test]
    fn add_scalar() {
        let a = tensor([3], [1, 2, 3]);
        let c = a + 1;
        assert_eq!(c.data, [2, 3, 4]);
    }

    #[test]
    fn sub_scalar() {
        let a = tensor([3], [5, 6, 7]);
        let c = a - 2;
        assert_eq!(c.data, [3, 4, 5]);
    }

    #[test]
    fn mul_scalar() {
        let a = tensor([3], [1, 2, 3]);
        let c = a * 3;
        assert_eq!(c.data, [3, 6, 9]);
    }

    #[test]
    fn div_scalar() {
        let a = tensor([3], [4, 6, 8]);
        let c = a / 2;
        assert_eq!(c.data, [2, 3, 4]);
    }

    #[test]
    fn tensor_2d_ops() {
        let mut a = tensor([2, 2], [1, 2, 3, 4]);
        let b = tensor([2, 2], [5, 6, 7, 8]);
        a += &b;
        assert_eq!(a.data, [6, 8, 10, 12]);
    }

    #[test]
    fn data_shape_preservation() {
        let a = tensor([3], [1, 2, 3]);
        let b = tensor([3], [4, 5, 6]);
        let c = a.clone() + b.clone();
        assert_eq!(c.shape, a.shape);
        assert_eq!(c.shape, b.shape);
    }

    #[test]
    fn batched_matmul_simple() {
        // Shape: [2, 2, 3] (2 batches, 2 rows, 3 cols)
        let a_data = [
            1, 2, 3, 4, 5, 6, // batch 2
            7, 8, 9, 10, 11, 12,
        ];
        let a = ArrTensor::with_data([2, 2, 3], a_data);

        // Shape: [2, 3, 2] (2 batches, 3 rows, 2 cols)
        let b_data = [
            1, 2, 3, 4, 5, 6, // batch 2
            7, 8, 9, 10, 11, 12,
        ];
        let b = ArrTensor::with_data([2, 3, 2], b_data);

        let mut out: ArrTensor<i32, 8, 3> = ArrTensor::new([2, 2, 2]);

        a.matmul(&b, &mut out);

        let expected = [22, 28, 49, 64, 220, 244, 301, 334];

        assert_eq!(out.data(), expected);
    }
}
