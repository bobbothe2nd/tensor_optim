use crate::TensorOps;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

/// A tensor-like structure that owns no data and holds only slices.
///
/// The shape is `&'a [usize]`, meaning the shape can never be changed once instantiated.
///
/// The data is `&'a mut [data]`, allowing for the data to be changed at runtime.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RefTensor<'a, T> {
    shape: &'a [usize],
    data: &'a mut [T],
}

impl<'a, T> RefTensor<'a, T> {
    /// Constructs a new `RefTensor` borrowing mutable data slice with given static shape.
    ///
    /// Unlike owned `Tensor` types, `RefTensor` has no default constructor.
    /// If it did, the constructor would refuse to compile because slices have no statically known size.
    ///
    /// # Panics
    ///
    /// This constructor panics when the product of each dimension is not equal to the length of all the data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensor_optim::RefTensor;
    ///
    /// const SHAPE: &[usize] = &[2, 3];
    ///
    /// let mut data = [0f64; 6];
    /// let mut tensor = RefTensor::with_data(SHAPE, &mut data);
    ///
    /// tensor[&[1, 2]] = 42.0;
    /// println!("First element: {}", tensor[&[0, 0]]);
    /// ```
    #[must_use]
    pub fn with_data(shape: &'a [usize], data: &'a mut [T]) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape and data length mismatch"
        );
        Self { shape, data }
    }
}

impl<T> TensorOps<T> for RefTensor<'_, T> {
    fn data(&self) -> &[T] {
        self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    fn shape(&self) -> &[usize] {
        self.shape
    }
}

impl<T> Index<&[usize]> for RefTensor<'_, T> {
    type Output = T;
    fn index(&self, idx: &[usize]) -> &Self::Output {
        let flat = self
            .index_offset(idx)
            .expect("recieved invalid index into Reftensor");
        &self.data[flat]
    }
}

impl<T> IndexMut<&[usize]> for RefTensor<'_, T> {
    fn index_mut(&mut self, idx: &[usize]) -> &mut Self::Output {
        let flat = self
            .index_offset(idx)
            .expect("recieved invalid index into Reftensor");
        &mut self.data[flat]
    }
}

impl<T> AddAssign<&Self> for RefTensor<'_, T>
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

impl<T> SubAssign<&Self> for RefTensor<'_, T>
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

impl<T> MulAssign<&Self> for RefTensor<'_, T>
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

impl<T> DivAssign<&Self> for RefTensor<'_, T>
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

impl<T> AddAssign<T> for RefTensor<'_, T>
where
    T: Copy + Add<Output = T>,
{
    fn add_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a = *a + rhs;
        }
    }
}

impl<T> SubAssign<T> for RefTensor<'_, T>
where
    T: Copy + Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a = *a - rhs;
        }
    }
}

impl<T> MulAssign<T> for RefTensor<'_, T>
where
    T: Copy + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a = *a * rhs;
        }
    }
}

impl<T> DivAssign<T> for RefTensor<'_, T>
where
    T: Copy + Div<Output = T>,
{
    fn div_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a = *a / rhs;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::internal::views::RefTensor;

    #[test]
    fn index_immutable() {
        let shape: &[usize] = &[2, 2];

        let mut data = [0f64; 4];

        let mut t = RefTensor::with_data(shape, &mut data);
        let num = t[&[1, 0]];
        t[&[1, 0]] = 2.0;
        let new = t[&[1, 0]];
        assert_ne!(num, new);
        assert_eq!(new, 2.0);
    }

    fn tensor<'a, T>(shape: &'static [usize], buf: &'a mut [T]) -> RefTensor<'a, T> {
        RefTensor::with_data(shape, buf)
    }

    #[test]
    fn add_assign_tensor() {
        let mut a_buf = [1.0, 2.0, 3.0];
        let mut b_buf = [4.0, 5.0, 6.0];
        let mut a = tensor(&[3], &mut a_buf);
        let b = tensor(&[3], &mut b_buf);
        a += &b;
        assert_eq!(a.data, &[5.0, 7.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_assign_shape_mismatch() {
        let mut a_buf = [1.0, 2.0, 3.0];
        let mut b_buf = [4.0, 5.0];
        let mut a = tensor(&[3], &mut a_buf);
        let b = tensor(&[2], &mut b_buf);
        a += &b;
    }

    #[test]
    fn sub_assign_tensor() {
        let mut a_buf = [5.0, 7.0, 9.0];
        let mut b_buf = [4.0, 5.0, 6.0];
        let mut a = tensor(&[3], &mut a_buf);
        let b = tensor(&[3], &mut b_buf);
        a -= &b;
        assert_eq!(a.data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn mul_assign_scalar() {
        let mut a_buf = [1.0, 2.0, 3.0];
        let mut a = tensor(&[3], &mut a_buf);
        a *= 2.0;
        assert_eq!(a.data, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn div_assign_scalar() {
        let mut a_buf = [2.0, 4.0, 6.0];
        let mut a = tensor(&[3], &mut a_buf);
        a /= 2.0;
        assert_eq!(a.data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn mul_assign_tensor() {
        let mut a_buf = [1.0, 2.0, 3.0];
        let mut b_buf = [4.0, 5.0, 6.0];
        let mut a = tensor(&[3], &mut a_buf);
        let b = tensor(&[3], &mut b_buf);
        a *= &b;
        assert_eq!(a.data, &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn div_assign_tensor() {
        let mut a_buf = [4.0, 10.0, 18.0];
        let mut b_buf = [4.0, 5.0, 6.0];
        let mut a = tensor(&[3], &mut a_buf);
        let b = tensor(&[3], &mut b_buf);
        a /= &b;
        assert_eq!(a.data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn basic_add_assign() {
        let mut a_data = [1, 2, 3];
        let mut b_data = [4, 5, 6];
        let mut a = tensor(&[3], &mut a_data);
        let b = tensor(&[3], &mut b_data);
        a += &b;
        assert_eq!(a.data, &[5, 7, 9]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn mismatched_shapes_panics() {
        let mut a_buf = [1, 2];
        let mut b_buf = [3, 4, 5];
        let mut a = tensor(&[2], &mut a_buf);
        let b = tensor(&[3], &mut b_buf);
        a += &b; // should panic
    }
}
