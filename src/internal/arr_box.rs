use super::{ConstTensorOps, TensorOps};
use core::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};
use lazy_simd::{
    scalar::Primitive,
    simd::{backend::AlignedSimd, LaneCount, Simd, SimdElement, SupportedLaneCount},
    MAX_SIMD_SINGLE_PRECISION_LANES,
};

#[cfg(feature = "no_stack")]
use alloc::boxed::Box;

/// A tensor made up of statically sized arrays.
///
/// Often the best choice for embedded tensor operations because it doesn't use any OS-dependent features like heap allocators.
/// If memory efficiency is the largest concern, the lack of dynamic heap allocation is a huge positive of `ArrTensor`.
///
/// However, when flexibility is put before memory efficiency and performance, this becomes obsolete; use `DynTensor` instead.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArrTensor<
    T,
    const N: usize,
    const D: usize,
    const LANES: usize = MAX_SIMD_SINGLE_PRECISION_LANES,
> where
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    shape: [usize; D],
    data: Box<Simd<T, N, LANES>>, // vector instead of array
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
        debug_assert_eq!(
            shape.iter().product::<usize>(),
            N,
            "shape and data length mismatch"
        );
        let data = {
            let mut uninit = Box::<Simd<T, N, LANES>>::new_uninit();
            let ptr = uninit.as_mut_ptr();
            unsafe {
                ptr.write_bytes(0, 1);
                uninit.assume_init()
            }
        };
        Self {
            shape,
            data,
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
    /// println!("first element: {}", tensor[&[0, 0]]);
    /// ```
    #[must_use]
    pub fn with_data(shape: [usize; D], data: [T; N]) -> Self {
        debug_assert_eq!(
            shape.iter().product::<usize>(),
            N,
            "shape and data length mismatch"
        );
        Self {
            shape,
            data: Box::new(Simd::new(data)),
        }
    }

    /// Instantiates a new Tensor with pre-allocated data.
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
    /// let data = Box::new([0f64; 6]);
    /// let mut tensor = ArrTensor::box_data(SHAPE, data);
    ///
    /// tensor += 42.0;
    /// println!("first element: {}", tensor[&[0, 0]]);
    /// ```
    #[must_use]
    pub fn box_data(shape: [usize; D], data: Box<[T; N]>) -> Self {
        debug_assert_eq!(
            shape.iter().product::<usize>(),
            N,
            "shape and data length mismatch"
        );
        Self {
            shape,
            data: Simd::new_boxed(data),
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
        F: FnMut(&T) -> U,
        U: SimdElement + Primitive,
        [U; LANES]: AlignedSimd<[U; LANES], U, { LANES }>,
    {
        let mut out = ArrTensor::new(self.shape);
        for (x, y) in out.data.iter_mut().zip(self.data.iter()) {
            *x = f(y);
        }
        out
    }

    /// Apply a function pairwise elementwise over two `ArrTensor`s, mapping to a new tensor.
    ///
    /// # Panics
    ///
    /// Both tensors, `self` and `other`, must have the same shape or a panic will occur.
    pub fn zip_map<U, V, F>(
        &self,
        other: &ArrTensor<U, N, D, LANES>,
        mut f: F,
    ) -> ArrTensor<V, N, D, LANES>
    where
        U: SimdElement + Primitive,
        [U; LANES]: AlignedSimd<[U; LANES], U, { LANES }>,
        V: SimdElement + Primitive,
        [V; LANES]: AlignedSimd<[V; LANES], V, { LANES }>,
        F: FnMut(&T, &U) -> V,
    {
        debug_assert_eq!(self.shape, other.shape, "shape mismatch in `zip_map`");

        let mut out = ArrTensor::new(self.shape);
        for (x, (y, z)) in out.data.iter_mut().zip(self.data.iter().zip(other.data.iter())) {
            *x = f(y, z);
        }
        out
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> TensorOps<T>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn data(&self) -> &[T] {
        &**self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut **self.data
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> ConstTensorOps<T, N, D>
    for ArrTensor<T, N, D, LANES>
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

impl<T, const N: usize, const D: usize, const LANES: usize> Index<&[usize]>
    for ArrTensor<T, N, D, LANES>
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

impl<T, const N: usize, const D: usize, const LANES: usize> AddAssign<&Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in AddAssign");
        *self.data += &*rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> AddAssign<Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in AddAssign");
        *self.data += *rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> AddAssign<T>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: T) {
        *self.data += rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> SubAssign<&Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in SubAssign");
        *self.data -= &*rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> SubAssign<Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in SubAssign");
        *self.data -= *rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> SubAssign<T>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: T) {
        *self.data -= rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> MulAssign<T>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: T) {
        *self.data *= rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> MulAssign<&Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in MulAssign");
        *self.data *= &*rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> MulAssign<Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in MulAssign");
        *self.data *= *rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> DivAssign<T>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: T) {
        *self.data /= rhs;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> DivAssign<&Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in DivAssign");
        *self.data /= &*rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> DivAssign<Self>
    for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in DivAssign");
        *self.data /= *rhs.data;
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Add<Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Add");
        self += rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Add<&Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Add");
        self += rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Add<T> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn add(mut self, rhs: T) -> Self {
        self += rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Sub<Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Sub");
        self -= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Sub<&Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Sub");
        self -= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Sub<T> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self {
        self -= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Mul<Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Mul");
        self *= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Mul<&Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(mut self, rhs: &Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Mul");
        self *= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Mul<T> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self {
        self *= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Div<Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Div");
        self /= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Div<&Self> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn div(mut self, rhs: &Self) -> Self {
        debug_assert_eq!(self.shape, rhs.shape, "shape mismatch in Div");
        self /= rhs;
        self
    }
}

impl<T, const N: usize, const D: usize, const LANES: usize> Div<T> for ArrTensor<T, N, D, LANES>
where
    T: SimdElement + Primitive + Default,
    [T; LANES]: AlignedSimd<[T; LANES], T, { LANES }>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn div(mut self, rhs: T) -> Self {
        self /= rhs;
        self
    }
}

fn compute_strides_fixed<const D: usize>(shape: &[usize], out: &mut [usize; D]) {
    let mut stride = 1;
    for i in (0..D).rev() {
        out[i] = stride;
        stride *= shape[i];
    }
}

fn unravel_index_fixed<const D: usize>(
    mut idx: usize,
    shape: &[usize],
    out: &mut [usize; D],
) {
    for i in (0..D - 2).rev() {
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
    /// Matrix multiplication cannot be performed on `ArrTensor`s when the dimensions exceed `D`.
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
            debug_assert!(
                D >= 2 && D <= D,
                "rank must be >=2 and <= D"
            );
        }

        let m = self.shape[D - 2];
        let k = self.shape[D - 1];
        let n = rhs.shape[D - 1];
        if n == 0 {
            return;
        }

        debug_assert!(k == rhs.shape[D - 2], "inner dimensions must match");
        debug_assert!(
            self.shape[..D - 2] == rhs.shape[..D - 2],
            "batch dimensions must match"
        );
        debug_assert!(
            self.shape[..D - 2] == out.shape[..D - 2],
            "batch dimensions must match output"
        );

        out.data.fill(T::default());

        // compute strides once
        let mut self_strides = [0usize; D];
        let mut rhs_strides = [0usize; D];
        let mut out_strides = [0usize; D];
        compute_strides_fixed(&self.shape, &mut self_strides);
        compute_strides_fixed(&rhs.shape, &mut rhs_strides);
        compute_strides_fixed(&out.shape, &mut out_strides);

        let batch_count = self.shape[..D - 2].iter().product::<usize>();
        if batch_count == 0 {
            return;
        }

        let mut batch_multi_idx = [0usize; D];

        for batch_idx in 0..batch_count {
            // unravel batch index into multi-index
            unravel_index_fixed(batch_idx, &self.shape, &mut batch_multi_idx);

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
            debug_assert!(
                D >= 2 && D <= D,
                "rank must be >=2 and <= D"
            );
        }

        let m = self.shape[D - 2];
        let k = self.shape[D - 1];
        let n = rhs.shape[D - 1];
        if n == 0 {
            return;
        }

        debug_assert!(k == rhs.shape[D - 2], "inner dimensions must match");
        debug_assert!(
            self.shape[..D - 2] == rhs.shape[..D - 2],
            "batch dimensions must match"
        );
        debug_assert!(
            self.shape[..D - 2] == out.shape[..D - 2],
            "batch dimensions must match output"
        );

        out.data.fill(0.0);

        // compute strides once
        let mut self_strides = [0usize; D];
        let mut rhs_strides = [0usize; D];
        let mut out_strides = [0usize; D];
        compute_strides_fixed(&self.shape, &mut self_strides);
        compute_strides_fixed(&rhs.shape, &mut rhs_strides);
        compute_strides_fixed(&out.shape, &mut out_strides);

        let batch_count = self.shape[..D - 2].iter().product::<usize>();
        if batch_count == 0 {
            return;
        }

        let mut batch_multi_idx = [0usize; D];

        for batch_idx in 0..batch_count {
            // unravel batch index into multi-index
            unravel_index_fixed(batch_idx, &self.shape, &mut batch_multi_idx);

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
                let out_row = &mut out.data[out_row_offset..(out_row_offset + n)];

                for kk in 0..k {
                    let a = self.data[self_row_offset + kk];
                    let rhs_row_offset = rhs_batch_offset + kk * rhs_strides[D - 2];
                    let rhs_row = &rhs.data[rhs_row_offset..(rhs_row_offset + n)];

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
            debug_assert!(
                D >= 2 && D <= D,
                "rank must be >=2 and <= D"
            );
        }

        let m = self.shape[D - 2];
        let k = self.shape[D - 1];
        let n = rhs.shape[D - 1];
        if n == 0 {
            return;
        }

        debug_assert!(k == rhs.shape[D - 2], "inner dimensions must match");
        debug_assert!(
            self.shape[..D - 2] == rhs.shape[..D - 2],
            "batch dimensions must match"
        );
        debug_assert!(
            self.shape[..D - 2] == out.shape[..D - 2],
            "batch dimensions must match output"
        );

        out.data.fill(0.0);

        // compute strides once
        let mut self_strides = [0usize; D];
        let mut rhs_strides = [0usize; D];
        let mut out_strides = [0usize; D];
        compute_strides_fixed(&self.shape, &mut self_strides);
        compute_strides_fixed(&rhs.shape, &mut rhs_strides);
        compute_strides_fixed(&out.shape, &mut out_strides);

        let batch_count = self.shape[..D - 2].iter().product::<usize>();
        if batch_count == 0 {
            return;
        }

        let mut batch_multi_idx = [0usize; D];

        for batch_idx in 0..batch_count {
            // unravel batch index into multi-index
            unravel_index_fixed(batch_idx, &self.shape, &mut batch_multi_idx);

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
                let out_row = &mut out.data[out_row_offset..(out_row_offset + n)];

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
    /// Panics if `D` exceeds `D`.
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
            // rWeverse axes for ranks > 2
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
                debug_assert!(p < D, "transpose_axes: invalid axis in permutation");
                debug_assert!(!check[p], "transpose_axes: duplicate axis in permutation");
                check[p] = true;
            }
        }

        self.transpose_axes_unchecked(perm)
    }

    /// Permutes the axes of `self` assuming a valid permutation.
    ///
    /// This is roughly equivalent, though marginally more efficient, compared
    /// to [`Self::transpose_axes`].
    #[must_use]
    pub fn transpose_axes_unchecked(&self, perm: [usize; D]) -> Self {
        // compute new shape by permuting old shape
        let mut new_shape = [0usize; D];
        for i in 0..D {
            new_shape[i] = self.shape[perm[i]];
        }

        // compute old strides and new strides
        let mut old_strides = [0usize; D];
        let mut new_strides = [0usize; D];
        compute_strides_fixed(&self.shape, &mut old_strides);
        compute_strides_fixed(&new_shape, &mut new_strides);

        // allocate new data array
        let mut new_data = Box::<Simd<T, N, LANES>>::new_uninit();

        // for every flat index in new_data, find corresponding index in self.data
        for new_flat_index in 0..N {
            // unravel new_flat_index to multi-dim index in permuted axes
            let mut new_multi_index = [0usize; D];
            unravel_index_fixed(new_flat_index, &new_shape, &mut new_multi_index);

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

            unsafe {
                new_data.as_mut_ptr().cast::<T>().add(new_flat_index).write(self.data[old_flat_index]);
            }
        }

        let new_data = unsafe { new_data.assume_init() };

        Self {
            shape: new_shape,
            data: new_data,
        }
    }
}
