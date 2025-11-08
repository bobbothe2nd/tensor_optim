use tensor_optim::{ArrTensor, DynTensor, TensorOps};

#[test]
fn arr_and_dyn_transpose_equivalence() {
    let shape = [2, 3];
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let arr_tensor = ArrTensor::with_data(shape, data);

    let dyn_tensor: DynTensor<f64> = DynTensor::with_data(&shape, &data);

    let arr_t = arr_tensor.transpose();
    let dyn_t = dyn_tensor.transpose();

    assert_eq!(arr_t.shape(), [3, 2]);
    assert_eq!(dyn_t.shape(), &[3, 2]);

    for r in 0..shape[0] {
        for c in 0..shape[1] {
            let original_val = data[r * shape[1] + c];
            let arr_val = arr_t[&[c, r]];
            let dyn_val = dyn_t[&[c, r]];
            assert_eq!(arr_val, original_val);
            assert_eq!(dyn_val, original_val);
        }
    }

    let arr_values: Vec<_> = arr_t.data().iter().cloned().collect();
    let dyn_values: Vec<_> = dyn_t.data().iter().cloned().collect();
    assert_eq!(arr_values, dyn_values);
}
