pub mod array;
pub mod views;

#[cfg(feature = "alloc")]
pub mod dynamic;

/// Operations even the simplest tensors need to get indexing, shape, and other content.
pub trait TensorOps<T> {
    /// Provides the shape of the current tensor as a slice.
    ///
    /// Note: the builtin `index_offset` method will work immediately if the shape is stored row-major.
    fn shape(&self) -> &[usize];

    /// Provides the data of the current tensor in a slice of a generic `T`.
    ///
    /// Note: the builtin `index_offset` method will work immediately if the data is stored row-major.
    fn data(&self) -> &[T];

    /// Provides the data of the current tensor in a slice of a generic `T`.
    ///
    /// Note: the builtin `index_offset` method will work immediately if the data is stored row-major.
    fn data_mut(&mut self) -> &mut [T];

    /// Finds the index in `data` given a slice with one entry/dimension.
    fn index_offset(&self, idx: &[usize]) -> Option<usize> {
        if idx.len() != self.shape().len() {
            return None;
        }
        let mut stride = 1;
        let mut flat_index = 0;
        for (&i, &dim) in idx.iter().rev().zip(self.shape().iter().rev()) {
            if i >= dim {
                return None; // out of bounds
            }
            flat_index += i * stride;
            stride *= dim;
        }
        Some(flat_index)
    }
}

/// Operations only statically sized, non-allocating tensors can leverage.
pub trait ConstTensorOps<T, const N: usize, const D: usize> {
    /// Provides the shape of the current tensor as an array.
    ///
    /// Note: the builtin `index_offset` method will work immediately if the shape is stored row-major.
    fn shape_array(&self) -> &[usize; D];

    /// Provides the data of the current tensor in a slice of a generic `T`.
    ///
    /// Note: the builtin `index_offset` method will work immediately if the data is stored row-major.
    fn data_array(&self) -> &[T; N];

    /// Provides the data of the current tensor in a slice of a generic `T`.
    ///
    /// Note: the builtin `index_offset` method will work immediately if the data is stored row-major.
    fn data_mut_array(&mut self) -> &mut [T; N];
}

/// A small limit to ranks mostly in `ArrTensor`.
/// Primarily for matrix multiplication.
pub const MAX_STATIC_RANK: usize = 12;
