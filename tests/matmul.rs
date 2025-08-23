#[cfg(feature = "alloc")]
use tensor_optim::{ArrTensor, DynTensor, TensorOps};

#[cfg(feature = "alloc")]
#[test]
fn arr_and_dyn_tensor_matmul_equivalence() {
    // Batched matmul shapes: [batch, M, K] * [batch, K, N] = [batch, M, N]
    let batch = 2;
    let m = 3;
    let k = 4;
    let n = 5;

    let shape_a = [batch, m, k];
    let shape_b = [batch, k, n];
    let shape_out = [batch, m, n];

    let len_a = batch * m * k;
    let len_b = batch * k * n;
    let len_out = batch * m * n;

    // Generate test data (float32)
    let data_a: Vec<f32> = (0..len_a).map(|x| x as f32 + 1.0).collect();
    let data_b: Vec<f32> = (0..len_b).map(|x| (x as f32 + 1.0) * 0.5).collect();

    // DynTensor construction
    let dyn_a = DynTensor::from_vec(&shape_a, data_a.clone());
    let dyn_b = DynTensor::from_vec(&shape_b, data_b.clone());
    let mut dyn_out = DynTensor::new(&shape_out);

    dyn_a.matmul(&dyn_b, &mut dyn_out);

    // ArrTensor construction
    // This requires fixed arrays, so convert Vec to arrays using try_into.
    // N = len of data, D = rank = 3 for these tensors.

    let arr_data_a: [f32; 24] = data_a
        .as_slice()
        .try_into()
        .expect("Wrong length for ArrTensor A");
    let arr_data_b: [f32; 40] = data_b
        .as_slice()
        .try_into()
        .expect("Wrong length for ArrTensor B");

    let arr_a = ArrTensor::<f32, 24, 3>::with_data(shape_a, arr_data_a);
    let arr_b = ArrTensor::<f32, 40, 3>::with_data(shape_b, arr_data_b);
    let mut arr_out = ArrTensor::<f32, 30, 3>::new(shape_out);

    arr_a.matmul(&arr_b, &mut arr_out);

    // Compare output data elementwise
    for i in 0..len_out {
        let dval = dyn_out.data()[i];
        let aval = arr_out.data()[i];
        let diff = (dval - aval).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: DynTensor={} ArrTensor={} (diff {})",
            i,
            dval,
            aval,
            diff
        );
    }
}
