use lazy_simd::{MAX_SIMD_SINGLE_PRECISION_LANES, scalar::{AddByRef, DivByRef, MulByRef, Primitive, SubByRef}, simd::{
    LaneCount, Simd, SimdElement, SupportedLaneCount, backend::AlignedSimd
}};
use crate::internal::{ConstTensorOps, TensorOps, };
use core::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};

/// A tensor made up of statically sized arrays.
///
/// Often the best choice for embedded tensor operations because it doesn't use any OS-dependent features like heap allocators.
/// If memory efficiency is the largest concern, the lack of dynamic heap allocation is a huge positive of `ArrTensor`.
///
/// However, when flexibility is put before memory efficiency and performance, this becomes obsolete; use `DynTensor` instead.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArrTensor<T, const N: usize, const D: usize, const LANES: usize = MAX_SIMD_SINGLE_PRECISION_LANES>
where 
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    shape: [usize; D],
    data: Simd<T, N, LANES>, // vector instead of array
}

impl<T, const N: usize, const D: usize, const LANES: usize> ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
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
            data: Simd::default(),
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
        Self {
            shape,
            data: Simd::new(data),
        }
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Map each element of `ArrTensor<T, N, D, LANES>` to `ArrTensor<U, N, D>` by applying `f` elementwise.
    pub fn map<U, F>(&self, mut f: F) -> ArrTensor<U, N, D, LANES>
    where
        F: FnMut(T) -> U,
        U: SimdElement + Primitive,
        [U; LANES]: AlignedSimd<[U; LANES], U, { LANES }>,
    {
        let new_data = core::array::from_fn(|i| f(self.data[i]));
        ArrTensor {
            shape: self.shape,
            data: Simd::new(new_data),
        }
    }

    /// Apply a function pairwise elementwise over two `ArrTensor`s, mapping to a new tensor.
    ///
    /// # Panics
    ///
    /// Both tensors, `self` and `other`, must have the same shape or a panic will occur.
    pub fn zip_map<U, V, F>(&self, other: &ArrTensor<U, N, D, LANES>, mut f: F) -> ArrTensor<V, N, D, LANES>
    where
        U: SimdElement + Primitive,
        [U; LANES]: AlignedSimd<[U; LANES], U, { LANES }>,
        V: SimdElement + Primitive,
        [V; LANES]: AlignedSimd<[V; LANES], V, { LANES }>,
        F: FnMut(T, U) -> V,
    {
        assert_eq!(self.shape, other.shape, "shape mismatch in zip_map");

        let new_data = core::array::from_fn(|i| f(self.data[i], other.data[i]));

        ArrTensor {
            shape: self.shape,
            data: Simd::new(new_data),
        }
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> TensorOps<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn data(&self) -> &[T] {
        &*self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut *self.data
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> ConstTensorOps<T, N, D> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
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

impl<T, const N: usize, const D: usize, const LANES: usize> Index<&[usize]> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = T;

    fn index(&self, idx: &[usize]) -> &Self::Output {
        let flat = self
            .index_offset(idx)
            .expect("recieved invalid index into tensor");
        &self.data[flat]
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> AddAssign<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data += rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> AddAssign<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data = self.data.as_add(&rhs.data);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> AddAssign<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: T) {
        self.data += rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> AddAssign<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: &T) {
        self.data = self.data.as_add(rhs);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> SubAssign<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data -= rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> SubAssign<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data = self.data.as_sub(&rhs.data);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> SubAssign<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: T) {
        self.data -= rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> SubAssign<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: &T) {
        self.data = self.data.as_sub(rhs);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> MulAssign<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: T) {
        self.data *= rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> MulAssign<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: &T) {
        self.data = self.data.as_mul(rhs);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> MulAssign<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data *= rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> MulAssign<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data = self.data.as_mul(&rhs.data);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> DivAssign<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: T) {
        self.data /= rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> DivAssign<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: &T) {
        self.data = self.data.as_div(rhs);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> DivAssign<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data /= rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> DivAssign<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape, "shape mismatch");
        self.data = self.data.as_div(&rhs.data);
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Add<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Add");
        let data = self.data.as_add(&rhs.data);
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Add<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Add");
        let data = self.data + rhs.data;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Add<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        let data = self.data + rhs;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Add<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn add(self, rhs: &T) -> Self {
        let data = self.data.as_add(rhs);
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Sub<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Sub");
        let data = self.data.as_sub(&rhs.data);
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Sub<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Sub");
        let data = self.data - rhs.data;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Sub<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        let data = self.data - rhs;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Sub<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, rhs: &T) -> Self {
        let data = self.data.as_sub(rhs);
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Mul<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Mul");
        let data = self.data.as_mul(&rhs.data);
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Mul<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Mul");
        let data = self.data * rhs.data;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Mul<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let data = self.data * rhs;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Mul<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: &T) -> Self {
        let data = self.data.as_mul(rhs);
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Div<&Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn div(self, rhs: &Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Div");
        let data = self.data.as_div(&rhs.data);
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Div<Self> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Div");
        let data = self.data / rhs.data;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Div<T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        let data = self.data / rhs;
        Self::with_data(self.shape, *data)
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Div<&T> for ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn div(self, rhs: &T) -> Self {
        let data = self.data.as_div(rhs);
        Self::with_data(self.shape, *data)
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

impl<T, const N: usize, const D: usize, const LANES: usize> ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
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
        rhs: &ArrTensor<T, M, D, LANES>,
        out: &mut ArrTensor<T, O, D, LANES>,
    ) {
        const {
            assert!(
                D >= 2 && D <= MAX_STATIC_RANK,
                "rank must be >=2 and <= MAX_STATIC_RANK"
            );
        }

        let m = self.shape[D - 2];
        let k = self.shape[D - 1];
        let n = rhs.shape[D - 1];
        if n == 0 {
            return;
        }

        assert!(k == rhs.shape[D - 2], "inner dimensions must match");
        assert!(
            self.shape[..D - 2] == rhs.shape[..D - 2],
            "batch dimensions must match"
        );
        assert!(
            self.shape[..D - 2] == out.shape[..D - 2],
            "batch dimensions must match output"
        );

        out.data.fill(T::default());

        // compute strides once
        let mut self_strides = [0usize; MAX_STATIC_RANK];
        let mut rhs_strides = [0usize; MAX_STATIC_RANK];
        let mut out_strides = [0usize; MAX_STATIC_RANK];
        compute_strides_fixed(&self.shape, &mut self_strides, D);
        compute_strides_fixed(&rhs.shape, &mut rhs_strides, D);
        compute_strides_fixed(&out.shape, &mut out_strides, D);

        let batch_count = self.shape[..D - 2].iter().product::<usize>();
        if batch_count == 0 {
            return;
        }

        let mut batch_multi_idx = [0usize; MAX_STATIC_RANK];

        for batch_idx in 0..batch_count {
            // unravel batch index into multi-index
            unravel_index_fixed(batch_idx, &self.shape[..D - 2], &mut batch_multi_idx, D - 2);

            // compute linear batch offsets
            let self_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&self_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();
            let rhs_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&rhs_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();
            let out_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&out_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();

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

impl<const N: usize, const D: usize> ArrTensor<f32, N, D> {
    /// SIMD-accelerated matrix multiplication like [`Self::matmul`].
    ///
    /// This is purely an enhanced version of regular matrix multiplication with the
    /// addition of SIMD acceleration. Implementation and details should be found
    /// consulting that documentation, not this function.
    ///
    /// # Panics
    ///
    /// Same preconditions as generic `matmul`.
    pub fn simd_matmul<const M: usize, const O: usize>(
        &self,
        rhs: &ArrTensor<f32, M, D>,
        out: &mut ArrTensor<f32, O, D>,
    ) {
        const {
            assert!(
                D >= 2 && D <= MAX_STATIC_RANK,
                "rank must be >=2 and <= MAX_STATIC_RANK"
            );
        }

        let m = self.shape[D - 2];
        let k = self.shape[D - 1];
        let n = rhs.shape[D - 1];
        if n == 0 {
            return;
        }

        assert!(k == rhs.shape[D - 2], "inner dimensions must match");
        assert!(
            self.shape[..D - 2] == rhs.shape[..D - 2],
            "batch dimensions must match"
        );
        assert!(
            self.shape[..D - 2] == out.shape[..D - 2],
            "batch dimensions must match output"
        );

        out.data.fill(0.0);

        // compute strides once
        let mut self_strides = [0usize; MAX_STATIC_RANK];
        let mut rhs_strides = [0usize; MAX_STATIC_RANK];
        let mut out_strides = [0usize; MAX_STATIC_RANK];
        compute_strides_fixed(&self.shape, &mut self_strides, D);
        compute_strides_fixed(&rhs.shape, &mut rhs_strides, D);
        compute_strides_fixed(&out.shape, &mut out_strides, D);

        let batch_count = self.shape[..D - 2].iter().product::<usize>();
        if batch_count == 0 {
            return;
        }

        let mut batch_multi_idx = [0usize; MAX_STATIC_RANK];

        for batch_idx in 0..batch_count {
            // unravel batch index into multi-index
            unravel_index_fixed(batch_idx, &self.shape[..D - 2], &mut batch_multi_idx, D - 2);

            // compute linear batch offsets
            let self_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&self_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();
            let rhs_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&rhs_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();
            let out_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&out_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();

            // linear inner matmul over last two dimensions
            for i in 0..m {
                let self_row_offset = self_batch_offset + i * self_strides[D - 2];
                let out_row_offset = out_batch_offset + i * out_strides[D - 2];
                let out_row = &mut out.data.as_mut_array()[out_row_offset..(out_row_offset + n)];

                for kk in 0..k {
                    let a = self.data[self_row_offset + kk];
                    let rhs_row_offset = rhs_batch_offset + kk * rhs_strides[D - 2];
                    let rhs_row = &rhs.data.as_array()[rhs_row_offset..(rhs_row_offset + n)];

                    lazy_simd::simd::mul_add_scalar_slice(a, rhs_row, out_row);
                }
            }
        }
    }
}

impl<const N: usize, const D: usize> ArrTensor<f64, N, D> {
    /// SIMD-accelerated single-precision matrix multiplication like [`Self::matmul`].
    ///
    /// This is purely an enhanced version of regular matrix multiplication with the
    /// addition of SIMD acceleration. Implementation and details should be found
    /// consulting that documentation, not this function.
    ///
    /// # Panics
    ///
    /// Same preconditions as generic `matmul`.
    pub fn simd_matmul<const M: usize, const O: usize>(
        &self,
        rhs: &ArrTensor<f64, M, D>,
        out: &mut ArrTensor<f64, O, D>,
    ) {
        const {
            assert!(
                D >= 2 && D <= MAX_STATIC_RANK,
                "rank must be >=2 and <= MAX_STATIC_RANK"
            );
        }

        let m = self.shape[D - 2];
        let k = self.shape[D - 1];
        let n = rhs.shape[D - 1];
        if n == 0 {
            return;
        }

        assert!(k == rhs.shape[D - 2], "inner dimensions must match");
        assert!(
            self.shape[..D - 2] == rhs.shape[..D - 2],
            "batch dimensions must match"
        );
        assert!(
            self.shape[..D - 2] == out.shape[..D - 2],
            "batch dimensions must match output"
        );

        out.data.fill(0.0);

        // compute strides once
        let mut self_strides = [0usize; MAX_STATIC_RANK];
        let mut rhs_strides = [0usize; MAX_STATIC_RANK];
        let mut out_strides = [0usize; MAX_STATIC_RANK];
        compute_strides_fixed(&self.shape, &mut self_strides, D);
        compute_strides_fixed(&rhs.shape, &mut rhs_strides, D);
        compute_strides_fixed(&out.shape, &mut out_strides, D);

        let batch_count = self.shape[..D - 2].iter().product::<usize>();
        if batch_count == 0 {
            return;
        }

        let mut batch_multi_idx = [0usize; MAX_STATIC_RANK];

        for batch_idx in 0..batch_count {
            // unravel batch index into multi-index
            unravel_index_fixed(batch_idx, &self.shape[..D - 2], &mut batch_multi_idx, D - 2);

            // compute linear batch offsets
            let self_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&self_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();
            let rhs_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&rhs_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();
            let out_batch_offset: usize = batch_multi_idx[..D - 2]
                .iter()
                .zip(&out_strides[..D - 2])
                .map(|(&i, &s)| i * s)
                .sum();

            // linear inner matmul over last two dimensions
            for i in 0..m {
                let self_row_offset = self_batch_offset + i * self_strides[D - 2];
                let out_row_offset = out_batch_offset + i * out_strides[D - 2];
                let out_row = &mut out.data.as_mut_array()[out_row_offset..(out_row_offset + n)];

                for kk in 0..k {
                    let a = self.data[self_row_offset + kk];
                    let rhs_row_offset = rhs_batch_offset + kk * rhs_strides[D - 2];
                    let rhs_row = &rhs.data[rhs_row_offset..(rhs_row_offset + n)];

                    lazy_simd::simd::mul_add_scalar_slice_double(a, rhs_row, out_row);
                }
            }
        }
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> ArrTensor<T, N, D, LANES>
where 
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
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
    /// let tensor = ArrTensor::<f32, 6, 2>::with_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let transposed = tensor.transpose();
    /// assert_eq!(transposed.shape(), [3, 2]);
    /// assert_eq!(transposed.data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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

        self.transpose_axes_unchecked(perm)
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
    /// let tensor = ArrTensor::<f32, 6, 2>::with_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let transposed = tensor.transpose_axes([1, 0]);
    /// assert_eq!(transposed.shape(), [3, 2]);
    /// assert_eq!(transposed.data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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

        self.transpose_axes_unchecked(perm)
    }

    /// Permutes the axes of `self` assuming a valid permutation.
    ///
    /// This is roughly equivalent, though marginally more efficient, compared
    /// to [`Self::transpose_axes`].
    pub fn transpose_axes_unchecked(&self, perm: [usize; D]) -> Self {
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
            data: Simd::new(new_data),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{ArrTensor, TensorOps};

    #[test]
    fn add_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a += &b;
        assert_eq!(a.data.to_array(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [5.0, 7.0, 9.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a -= &b;
        assert_eq!(a.data.to_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn mul_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a *= &b;
        assert_eq!(a.data.to_array(), [4.0, 10.0, 18.0]);
    }

    #[test]
    fn div_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [4.0, 10.0, 18.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a /= &b;
        assert_eq!(a.data.to_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        a += 1.0;
        assert_eq!(a.data.to_array(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn sub_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [5.0, 6.0, 7.0]);
        a -= 2.0;
        assert_eq!(a.data.to_array(), [3.0, 4.0, 5.0]);
    }

    #[test]
    fn mul_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        a *= 3.0;
        assert_eq!(a.data.to_array(), [3.0, 6.0, 9.0]);
    }

    #[test]
    fn div_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [4.0, 6.0, 8.0]);
        a /= 2.0;
        assert_eq!(a.data.to_array(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn add_tensor() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c.data.to_array(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_tensor() {
        let a = ArrTensor::with_data([3], [5.0, 7.0, 9.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a - b;
        assert_eq!(c.data.to_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn mul_tensor() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a * b;
        assert_eq!(c.data.to_array(), [4.0, 10.0, 18.0]);
    }

    #[test]
    fn div_tensor() {
        let a = ArrTensor::with_data([3], [4.0, 10.0, 18.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a / b;
        assert_eq!(c.data.to_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_scalar() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let c = a + 1f64;
        assert_eq!(c.data.to_array(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn sub_scalar() {
        let a = ArrTensor::with_data([3], [5.0, 6.0, 7.0]);
        let c = a - 2f64;
        assert_eq!(c.data.to_array(), [3.0, 4.0, 5.0]);
    }

    #[test]
    fn mul_scalar() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let c = a * 3f64;
        assert_eq!(c.data.to_array(), [3.0, 6.0, 9.0]);
    }

    #[test]
    fn div_scalar() {
        let a = ArrTensor::with_data([3], [4f64, 6.0, 8.0]);
        let c = a / 2.0;
        assert_eq!(c.data.to_array(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn tensor_2d_ops() {
        let mut a = ArrTensor::with_data([2, 2], [1.0, 2.0, 3.0, 4.0]);
        let b = ArrTensor::with_data([2, 2], [5f64, 6.0, 7.0, 8.0]);
        a += &b;
        assert_eq!(a.data.to_array(), [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn data_shape_preservation() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a.clone() + b.clone();
        assert_eq!(c.shape, a.shape);
        assert_eq!(c.shape, b.shape);
    }

    #[test]
    fn batched_matmul_simple() {
        // shape: [2, 2, 3] (2 batches, 2 rows, 3 cols)
        let a_data = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 2
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let a = ArrTensor::with_data([2, 2, 3], a_data);

        // shape: [2, 3, 2] (2 batches, 3 rows, 2 cols)
        let b_data = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 2
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let b = ArrTensor::with_data([2, 3, 2], b_data);

        let mut out: ArrTensor<f32, 8, 3> = ArrTensor::new([2, 2, 2]);

        let expected = [22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0];

        // first normal matmul
        a.matmul(&b, &mut out);

        assert_eq!(out.data(), expected);

        // then simd accelerated
        a.simd_matmul(&b, &mut out);

        assert_eq!(out.data(), expected);
    }
}
