#[cfg(feature = "no_stack")]
pub mod arr_box;
#[cfg(not(feature = "no_stack"))]
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

/// Operations only statically sized, non-allocating tensors can use.
pub trait ConstTensorOps<T, const N: usize, const D: usize>: TensorOps<T> {
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

#[cfg(test)]
mod tests {
    use crate::{ArrTensor, TensorOps};

    #[test]
    fn add_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a += b;
        assert_eq!(a.data(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [5.0, 7.0, 9.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a -= b;
        assert_eq!(a.data(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn mul_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a *= b;
        assert_eq!(a.data(), [4.0, 10.0, 18.0]);
    }

    #[test]
    fn div_assign_tensor() {
        let mut a = ArrTensor::with_data([3], [4.0, 10.0, 18.0]);
        let b = ArrTensor::with_data([3], [4.0, 5.0, 6.0]);
        a /= b;
        assert_eq!(a.data(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        a += 1.0;
        assert_eq!(a.data(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn sub_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [5.0, 6.0, 7.0]);
        a -= 2.0;
        assert_eq!(a.data(), [3.0, 4.0, 5.0]);
    }

    #[test]
    fn mul_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        a *= 3.0;
        assert_eq!(a.data(), [3.0, 6.0, 9.0]);
    }

    #[test]
    fn div_assign_scalar() {
        let mut a = ArrTensor::with_data([3], [4.0, 6.0, 8.0]);
        a /= 2.0;
        assert_eq!(a.data(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn add_tensor() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c.data(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_tensor() {
        let a = ArrTensor::with_data([3], [5.0, 7.0, 9.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a - b;
        assert_eq!(c.data(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn mul_tensor() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a * b;
        assert_eq!(c.data(), [4.0, 10.0, 18.0]);
    }

    #[test]
    fn div_tensor() {
        let a = ArrTensor::with_data([3], [4.0, 10.0, 18.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a / b;
        assert_eq!(c.data(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_scalar() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let c = a + 1f64;
        assert_eq!(c.data(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn sub_scalar() {
        let a = ArrTensor::with_data([3], [5.0, 6.0, 7.0]);
        let c = a - 2f64;
        assert_eq!(c.data(), [3.0, 4.0, 5.0]);
    }

    #[test]
    fn mul_scalar() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let c = a * 3f64;
        assert_eq!(c.data(), [3.0, 6.0, 9.0]);
    }

    #[test]
    fn div_scalar() {
        let a = ArrTensor::with_data([3], [4f64, 6.0, 8.0]);
        let c = a / 2.0;
        assert_eq!(c.data(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn tensor_2d_ops() {
        let mut a = ArrTensor::with_data([2, 2], [1.0, 2.0, 3.0, 4.0]);
        let b = ArrTensor::with_data([2, 2], [5f64, 6.0, 7.0, 8.0]);
        a += b;
        assert_eq!(a.data(), [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn data_shape_preservation() {
        let a = ArrTensor::with_data([3], [1.0, 2.0, 3.0]);
        let b = ArrTensor::with_data([3], [4f64, 5.0, 6.0]);
        let c = a.clone() + b.clone();
        assert_eq!(c.shape(), a.shape());
        assert_eq!(c.shape(), b.shape());
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
