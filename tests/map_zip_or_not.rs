use tensor_optim::{ArrTensor, DynTensor, TensorOps};

#[test]
fn map_and_zip_map_consistency() {
    let shape = [2, 2];
    let data = [1.0, 2.0, 3.0, 4.0];

    let arr_tensor = ArrTensor::with_data(shape, data);

    let dyn_tensor: DynTensor<f64> = DynTensor::with_data(&shape, &data);

    let arr_mapped = arr_tensor.map(|x| x * x);
    let dyn_mapped: DynTensor<f64> = dyn_tensor.map(|x| x * x);

    assert_eq!(arr_mapped.shape(), arr_tensor.shape());
    assert_eq!(dyn_mapped.shape(), &arr_tensor.shape()[..]);
    assert_eq!(arr_mapped.data(), dyn_mapped.data());

    let arr_tensor2 = ArrTensor::with_data(shape, [5.0, 6.0, 7.0, 8.0]);
    let dyn_tensor2: DynTensor<f64> = DynTensor::with_data(&shape, &[5.0, 6.0, 7.0, 8.0]);

    let arr_zipped = arr_tensor.zip_map(&arr_tensor2, |a, b| a + b);
    let dyn_zipped: DynTensor<f64> = dyn_tensor.zip_map(&dyn_tensor2, |a, b| a + b);

    assert_eq!(arr_zipped.shape(), arr_tensor.shape());
    assert_eq!(dyn_zipped.shape(), arr_tensor.shape());
    assert_eq!(arr_zipped.data(), dyn_zipped.data());
}
