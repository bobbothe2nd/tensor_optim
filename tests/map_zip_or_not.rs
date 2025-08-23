#[cfg(feature = "alloc")]
use tensor_optim::{ArrTensor, DynTensor, TensorOps};

#[cfg(feature = "alloc")]
// Helper to convert ArrTensor to Vec for comparison with DynTensor
fn arrtensor_to_vec<T: Copy, const N: usize, const D: usize>(t: &ArrTensor<T, N, D>) -> Vec<T> {
    t.data().to_vec()
}

#[cfg(feature = "alloc")]
#[test]
fn test_map_and_zip_map_consistency() {
    // Example: 2D tensor with 4 elements
    // For ArrTensor, N=4, D=2 (2D), shape=[2, 2]
    let shape = [2, 2];
    let arr_data = [1, 2, 3, 4];
    let dyn_data = vec![1, 2, 3, 4];

    // Create ArrTensor
    let arr_tensor = ArrTensor::with_data(shape, arr_data);

    // Create DynTensor
    let dyn_tensor = DynTensor::<i32>::from_vec(&shape, dyn_data);

    // map: square each element
    let arr_mapped = arr_tensor.map(|x| x * x);
    let dyn_mapped = dyn_tensor.map(|x| x * x);

    // Compare shapes and data
    assert_eq!(arr_mapped.shape(), arr_tensor.shape());
    assert_eq!(dyn_mapped.shape(), &arr_tensor.shape()[..]);
    assert_eq!(arrtensor_to_vec(&arr_mapped), dyn_mapped.data());

    // Create second tensor for zip_map
    let arr_tensor2 = ArrTensor::with_data(shape, [5, 6, 7, 8]);
    let dyn_tensor2 = DynTensor::<i32>::from_vec(&shape, vec![5, 6, 7, 8]);

    // zip_map: sum elements pairwise
    let arr_zipped = arr_tensor.zip_map(&arr_tensor2, |a, b| a + b);
    let dyn_zipped = dyn_tensor.zip_map(&dyn_tensor2, |a, b| a + b);

    // Compare shapes and data
    assert_eq!(arr_zipped.shape(), arr_tensor.shape());
    assert_eq!(dyn_zipped.shape(), &arr_tensor.shape()[..]);
    assert_eq!(arrtensor_to_vec(&arr_zipped), dyn_zipped.data());
}
