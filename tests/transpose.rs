#[cfg(feature = "alloc")]
use tensor_optim::{ArrTensor, DynTensor, TensorOps};

#[cfg(feature = "alloc")]
#[test]
fn arr_and_dyn_transpose_equivalence() {
    // Example: 2D matrix 2x3
    let shape = [2, 3];
    let data: [i32; 6] = [1, 2, 3, 4, 5, 6];

    // Static tensor
    let arr_tensor = ArrTensor::<i32, 6, 2>::with_data(shape, data);

    // Dynamic tensor
    let dyn_tensor = DynTensor::with_data(&shape, &data);

    // Transpose both
    let arr_t = arr_tensor.transpose();
    let dyn_t = dyn_tensor.transpose();

    // Expected shape after transpose: 3x2
    assert_eq!(arr_t.shape(), [3, 2]);
    assert_eq!(dyn_t.shape(), &[3, 2]);

    // Verify that all elements match
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            let original_val = data[r * shape[1] + c];
            let arr_val = arr_t[&[c, r]];
            let dyn_val = dyn_t[&[c, r]];
            assert_eq!(arr_val, original_val);
            assert_eq!(dyn_val, original_val);
        }
    }

    // Also check ArrTensor and DynTensor transposed values match each other
    let arr_values: Vec<_> = arr_t.data().iter().cloned().collect();
    let dyn_values: Vec<_> = dyn_t.data().iter().cloned().collect();
    assert_eq!(arr_values, dyn_values);
}
