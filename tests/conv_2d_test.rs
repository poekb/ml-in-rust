use cust::prelude::*;
use machine_learning_lib::init_cuda;
use machine_learning_lib::layers::dense::he_initializer;
use machine_learning_lib::layers::optimizer::{NoOpOptimizer, Optimizer};
use machine_learning_lib::layers::{Layer, conv_2d::Conv2DLayer};

#[test]
fn test_conv2d_forward() {
    let _ctx = init_cuda();

    // Create a small input: 1 channel, 3x3
    let input_depth = 1;
    let input_dim_x = 3;
    let input_dim_y = 3;

    // Kernel: 1 channel, 2x2 filter
    let kernel_dim_x = 2;
    let kernel_dim_y = 2;
    let kernel_count = 1;

    let optimizer: Box<dyn Optimizer> = NoOpOptimizer::boxed();

    // Create layer
    let mut layer = Conv2DLayer::new(
        input_depth,
        input_dim_x,
        input_dim_y,
        kernel_dim_x,
        kernel_dim_y,
        kernel_count,
        he_initializer,
        &optimizer,
    );

    // Replace random weights with known values for testing
    // Input: [
    //   1, 2, 3,
    //   4, 5, 6,
    //   7, 8, 9
    // ]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // Kernel: [
    //   1, 2,
    //   3, 4
    // ]
    let kernel = vec![1.0, 2.0, 3.0, 4.0];

    // Bias: [1.0]
    let bias = vec![1.0];

    // Override the layer's parameters with our test values
    let kernel_params = DeviceBuffer::from_slice(&kernel).unwrap();
    let bias_params = DeviceBuffer::from_slice(&bias).unwrap();

    // Reflection to set the private fields - in real test you would need to add methods to Conv2DLayer
    // to facilitate testing, or make these fields public for testing
    unsafe {
        let kernel_ptr =
            std::mem::transmute::<_, *mut DeviceBuffer<f32>>(&mut layer.kernel_parameters);
        *kernel_ptr = kernel_params;

        let bias_ptr = std::mem::transmute::<_, *mut DeviceBuffer<f32>>(&mut layer.bias_parameters);
        *bias_ptr = bias_params;
    }

    // Create device buffers for input and output
    let input_buffer = DeviceBuffer::from_slice(&input).unwrap();
    let mut output_buffer = DeviceBuffer::from_slice(&vec![
        0.0;
        (input_dim_x - kernel_dim_x + 1)
            * (input_dim_y - kernel_dim_y + 1)
            * kernel_count
    ])
    .unwrap();

    // Run forward pass
    layer
        .forward(&input_buffer, &mut output_buffer, None)
        .unwrap();

    // Get output back to host
    let mut output =
        vec![
            0.0;
            (input_dim_x - kernel_dim_x + 1) * (input_dim_y - kernel_dim_y + 1) * kernel_count
        ];
    output_buffer.copy_to(&mut output).unwrap();

    // Manually calculate expected output:
    // output[0,0] = 1*1 + 2*2 + 4*3 + 5*4 + 1(bias) = 1 + 4 + 12 + 20 + 1 = 38
    // output[0,1] = 2*1 + 3*2 + 5*3 + 6*4 + 1(bias) = 2 + 6 + 15 + 24 + 1 = 48
    // output[1,0] = 4*1 + 5*2 + 7*3 + 8*4 + 1(bias) = 4 + 10 + 21 + 32 + 1 = 68
    // output[1,1] = 5*1 + 6*2 + 8*3 + 9*4 + 1(bias) = 5 + 12 + 24 + 36 + 1 = 78
    let expected = vec![38.0, 48.0, 68.0, 78.0];

    // Check results
    assert_eq!(output.len(), expected.len());
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Output mismatch: {} vs {}",
            actual,
            expected
        );
    }
}

#[test]
fn test_conv2d_backward() {
    let _ctx = init_cuda();

    // Similar setup to forward test
    let input_depth = 1;
    let input_dim_x = 3;
    let input_dim_y = 3;
    let kernel_dim_x = 2;
    let kernel_dim_y = 2;
    let kernel_count = 1;

    let optimizer: Box<dyn Optimizer> = NoOpOptimizer::boxed();

    let mut layer = Conv2DLayer::new(
        input_depth,
        input_dim_x,
        input_dim_y,
        kernel_dim_x,
        kernel_dim_y,
        kernel_count,
        he_initializer,
        &optimizer,
    );

    // Input: [
    //   1, 2, 3,
    //   4, 5, 6,
    //   7, 8, 9
    // ]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // Kernel: [
    //   1, 2,
    //   3, 4
    // ]
    let kernel = vec![1.0, 2.0, 3.0, 4.0];

    layer.kernel_parameters = DeviceBuffer::from_slice(&kernel).unwrap();
    layer.bias_parameters = DeviceBuffer::from_slice(&vec![1.0]).unwrap();

    // Bias: [1.0]
    let bias = vec![1.0];

    // Output gradient: [
    //   0.1, 0.2,
    //   0.3, 0.4
    // ]
    let output_gradient = vec![0.1, 0.2, 0.3, 0.4];

    // Override layer parameters as in the forward test
    // ...

    // Create device buffers
    let input_buffer = DeviceBuffer::from_slice(&input).unwrap();
    let output_gradient_buffer = DeviceBuffer::from_slice(&output_gradient).unwrap();
    let mut input_gradient_buffer = DeviceBuffer::from_slice(&vec![0.0; input.len()]).unwrap();

    // Run backward pass
    layer
        .back_propagate(
            &mut input_gradient_buffer,
            &input_buffer,
            &output_gradient_buffer,
            None,
        )
        .unwrap();

    // Get results back to host
    let mut input_gradient = vec![0.0; input.len()];
    input_gradient_buffer.copy_to(&mut input_gradient).unwrap();

    // Get kernel and bias gradients
    let mut kernel_gradient = vec![0.0; kernel.len()];
    let mut bias_gradient = vec![0.0; bias.len()];

    // In real test you would need methods to access these or make them public for testing
    unsafe {
        let kernel_grad_ptr =
            std::mem::transmute::<_, *const DeviceBuffer<f32>>(&layer.kernel_gradient);
        kernel_grad_ptr
            .as_ref()
            .unwrap()
            .copy_to(&mut kernel_gradient)
            .unwrap();

        let bias_grad_ptr =
            std::mem::transmute::<_, *const DeviceBuffer<f32>>(&layer.bias_gradient);
        bias_grad_ptr
            .as_ref()
            .unwrap()
            .copy_to(&mut bias_gradient)
            .unwrap();
    }

    // Input gradient calculation:
    // input_grad[0,0] = output_grad[0,0] * kernel[0,0] = 0.1 * 1.0 = 0.1
    // input_grad[0,1] = output_grad[0,0] * kernel[0,1] + output_grad[0,1] * kernel[0,0] = 0.1 * 2.0 + 0.2 * 1.0 = 0.4

    // Correct expected input gradient values:
    let expected_input_gradient = vec![0.1, 0.4, 0.4, 0.6, 2.0, 1.6, 0.9, 2.4, 1.6];

    // Kernel gradient calculation:
    // kernel_grad[0,0] = output_grad[0,0] * input[0,0] + output_grad[0,1] * input[0,1] +
    //                     output_grad[1,0] * input[1,0] + output_grad[1,1] * input[1,1]
    // = 0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 4.0 + 0.4 * 5.0
    // = 0.1 + 0.4 + 1.2 + 2.0 = 3.7
    // ... (continue for all kernel positions)

    // Expected kernel gradient
    let expected_kernel_gradient = vec![3.7, 4.7, 6.7, 7.7];

    // Bias gradient calculation:
    // bias_grad[0] = sum of all output gradients = 0.1 + 0.2 + 0.3 + 0.4 = 1.0

    let expected_bias_gradient = vec![1.0];

    // Check results

    for (_, (actual, expected)) in bias_gradient
        .iter()
        .zip(expected_bias_gradient.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Bias gradient mismatch: {:?} vs {:?}",
            bias_gradient,
            expected_bias_gradient
        );
    }

    for (_, (actual, expected)) in kernel_gradient
        .iter()
        .zip(expected_kernel_gradient.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Kernel gradient mismatch: {:?} vs {:?}",
            kernel_gradient,
            expected_kernel_gradient
        );
    }

    for (_, (actual, expected)) in input_gradient
        .iter()
        .zip(expected_input_gradient.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Input gradient mismatch: {:?} vs {:?}",
            input_gradient,
            expected_input_gradient
        );
    }
}
