use cust::prelude::*;
use machine_learning_lib::init_cuda;
use machine_learning_lib::layers::dense::he_initializer;
use machine_learning_lib::layers::{Layer, conv_2d::Conv2DLayer};

#[test]
fn test_conv2d_multi_channel_forward() {
    let _ctx = init_cuda();

    // 2 input channels, 2x2 dimensions per channel
    let input_depth = 2;
    let input_dim_x = 2;
    let input_dim_y = 2;

    // 2x2 kernel, 2 output channels
    let kernel_dim_x = 2;
    let kernel_dim_y = 2;
    let kernel_count = 2;

    // Create layer
    let mut layer = Conv2DLayer::new(
        input_depth,
        input_dim_x,
        input_dim_y,
        kernel_dim_x,
        kernel_dim_y,
        kernel_count,
        he_initializer,
    );

    // Input: two 2x2 matrices
    // Channel 1: [1, 2, 3, 4]
    // Channel 2: [5, 6, 7, 8]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Kernel: two sets of 2x2 kernels (one set per output channel)
    // Kernel 1, Channel 1: [1, 2, 3, 4]
    // Kernel 1, Channel 2: [5, 6, 7, 8]
    // Kernel 2, Channel 1: [9, 10, 11, 12]
    // Kernel 2, Channel 2: [13, 14, 15, 16]
    let kernel = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    // Bias: one per output channel
    let bias = vec![0.5, 1.0];

    // Override the layer's parameters with our test values
    let kernel_params = DeviceBuffer::from_slice(&kernel).unwrap();
    let bias_params = DeviceBuffer::from_slice(&bias).unwrap();

    // Replace the random weights with our test values
    // You'll need to add methods to your Conv2DLayer for testing
    // or make these fields public for testing
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

    // Calculate expected output manually for both output channels:

    // For output channel 1 (first kernel set):
    // output[0][0][0] =
    //   input[0][0][0] * kernel[0][0][0][0] + input[0][0][1] * kernel[0][0][0][1] +
    //   input[0][1][0] * kernel[0][0][1][0] + input[0][1][1] * kernel[0][0][1][1] +
    //   input[1][0][0] * kernel[0][1][0][0] + input[1][0][1] * kernel[0][1][0][1] +
    //   input[1][1][0] * kernel[0][1][1][0] + input[1][1][1] * kernel[0][1][1][1] + bias[0]
    // = 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8 + 0.5
    // = 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 0.5 = 204.5

    // For output channel 2 (second kernel set):
    // output[1][0][0] =
    //   input[0][0][0] * kernel[1][0][0][0] + input[0][0][1] * kernel[1][0][0][1] +
    //   input[0][1][0] * kernel[1][0][1][0] + input[0][1][1] * kernel[1][0][1][1] +
    //   input[1][0][0] * kernel[1][1][0][0] + input[1][0][1] * kernel[1][1][0][1] +
    //   input[1][1][0] * kernel[1][1][1][0] + input[1][1][1] * kernel[1][1][1][1] + bias[1]
    // = 1*9 + 2*10 + 3*11 + 4*12 + 5*13 + 6*14 + 7*15 + 8*16 + 1.0
    // = 9 + 20 + 33 + 48 + 65 + 84 + 105 + 128 + 1.0 = 493.0

    let expected_output = vec![204.5, 493.0];

    // Verify the results
    assert_eq!(output.len(), expected_output.len());
    for (i, (actual, expected)) in output.iter().zip(expected_output.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Output mismatch at index {}: {} vs {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_conv2d_multi_channel_backward() {
    let _ctx = init_cuda();

    // Same setup as the forward test
    let input_depth = 2;
    let input_dim_x = 2;
    let input_dim_y = 2;
    let kernel_dim_x = 2;
    let kernel_dim_y = 2;
    let kernel_count = 2;

    let mut layer = Conv2DLayer::new(
        input_depth,
        input_dim_x,
        input_dim_y,
        kernel_dim_x,
        kernel_dim_y,
        kernel_count,
        he_initializer,
    );

    // Input: two 2x2 matrices
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Kernel: two sets of 2x2 kernels
    let kernel = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    // Bias: one per output channel
    let bias = vec![0.5, 1.0];

    // Output gradient: one value per output element (1x1 output size per channel)
    let output_gradient = vec![0.1, 0.2];

    // Override layer parameters
    let kernel_params = DeviceBuffer::from_slice(&kernel).unwrap();
    let bias_params = DeviceBuffer::from_slice(&bias).unwrap();

    unsafe {
        let kernel_ptr =
            std::mem::transmute::<_, *mut DeviceBuffer<f32>>(&mut layer.kernel_parameters);
        *kernel_ptr = kernel_params;

        let bias_ptr = std::mem::transmute::<_, *mut DeviceBuffer<f32>>(&mut layer.bias_parameters);
        *bias_ptr = bias_params;
    }

    // Create device buffers
    let input_buffer = DeviceBuffer::from_slice(&input).unwrap();
    let output_gradient_buffer = DeviceBuffer::from_slice(&output_gradient).unwrap();
    let mut input_gradient_buffer = DeviceBuffer::from_slice(&vec![0.0; input.len()]).unwrap();

    // Zero out kernel and bias gradients
    unsafe {
        let kernel_grad_ptr =
            std::mem::transmute::<_, *mut DeviceBuffer<f32>>(&mut layer.kernel_gradient);
        *kernel_grad_ptr = DeviceBuffer::from_slice(&vec![0.0; kernel.len()]).unwrap();

        let bias_grad_ptr =
            std::mem::transmute::<_, *mut DeviceBuffer<f32>>(&mut layer.bias_gradient);
        *bias_grad_ptr = DeviceBuffer::from_slice(&vec![0.0; bias.len()]).unwrap();
    }

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

    let mut kernel_gradient = vec![0.0; kernel.len()];
    let mut bias_gradient = vec![0.0; bias.len()];

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

    // Calculate expected gradients

    // Input gradients (applying the flipped kernel to the output gradient):
    // For input channel 1:
    // input_grad[0][0][0] = outgrad[0][0][0] * flipped_kernel[0][0][1][1] + outgrad[1][0][0] * flipped_kernel[1][0][1][1]
    //                      = 0.1 * 4 + 0.2 * 12 = 0.4 + 2.4 = 2.8
    // input_grad[0][0][1] = outgrad[0][0][0] * flipped_kernel[0][0][1][0] + outgrad[1][0][0] * flipped_kernel[1][0][1][0]
    //                      = 0.1 * 3 + 0.2 * 11 = 0.3 + 2.2 = 2.5
    // input_grad[0][1][0] = outgrad[0][0][0] * flipped_kernel[0][0][0][1] + outgrad[1][0][0] * flipped_kernel[1][0][0][1]
    //                      = 0.1 * 2 + 0.2 * 10 = 0.2 + 2.0 = 2.2
    // input_grad[0][1][1] = outgrad[0][0][0] * flipped_kernel[0][0][0][0] + outgrad[1][0][0] * flipped_kernel[1][0][0][0]
    //                      = 0.1 * 1 + 0.2 * 9 = 0.1 + 1.8 = 1.9

    // For input channel 2:
    // input_grad[1][0][0] = outgrad[0][0][0] * flipped_kernel[0][1][1][1] + outgrad[1][0][0] * flipped_kernel[1][1][1][1]
    //                      = 0.1 * 8 + 0.2 * 16 = 0.8 + 3.2 = 4.0
    // input_grad[1][0][1] = outgrad[0][0][0] * flipped_kernel[0][1][1][0] + outgrad[1][0][0] * flipped_kernel[1][1][1][0]
    //                      = 0.1 * 7 + 0.2 * 15 = 0.7 + 3.0 = 3.7
    // input_grad[1][1][0] = outgrad[0][0][0] * flipped_kernel[0][1][0][1] + outgrad[1][0][0] * flipped_kernel[1][1][0][1]
    //                      = 0.1 * 6 + 0.2 * 14 = 0.6 + 2.8 = 3.4
    // input_grad[1][1][1] = outgrad[0][0][0] * flipped_kernel[0][1][0][0] + outgrad[1][0][0] * flipped_kernel[1][1][0][0]
    //                      = 0.1 * 5 + 0.2 * 13 = 0.5 + 2.6 = 3.1

    let expected_input_gradient = vec![1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0];

    // Kernel gradients:
    // kernel_grad[0][0][0][0] = outgrad[0][0][0] * input[0][0][0] = 0.1 * 1 = 0.1
    // kernel_grad[0][0][0][1] = outgrad[0][0][0] * input[0][0][1] = 0.1 * 2 = 0.2
    // kernel_grad[0][0][1][0] = outgrad[0][0][0] * input[0][1][0] = 0.1 * 3 = 0.3
    // kernel_grad[0][0][1][1] = outgrad[0][0][0] * input[0][1][1] = 0.1 * 4 = 0.4
    // kernel_grad[0][1][0][0] = outgrad[0][0][0] * input[1][0][0] = 0.1 * 5 = 0.5
    // kernel_grad[0][1][0][1] = outgrad[0][0][0] * input[1][0][1] = 0.1 * 6 = 0.6
    // kernel_grad[0][1][1][0] = outgrad[0][0][0] * input[1][1][0] = 0.1 * 7 = 0.7
    // kernel_grad[0][1][1][1] = outgrad[0][0][0] * input[1][1][1] = 0.1 * 8 = 0.8

    // kernel_grad[1][0][0][0] = outgrad[1][0][0] * input[0][0][0] = 0.2 * 1 = 0.2
    // kernel_grad[1][0][0][1] = outgrad[1][0][0] * input[0][0][1] = 0.2 * 2 = 0.4
    // kernel_grad[1][0][1][0] = outgrad[1][0][0] * input[0][1][0] = 0.2 * 3 = 0.6
    // kernel_grad[1][0][1][1] = outgrad[1][0][0] * input[0][1][1] = 0.2 * 4 = 0.8
    // kernel_grad[1][1][0][0] = outgrad[1][0][0] * input[1][0][0] = 0.2 * 5 = 1.0
    // kernel_grad[1][1][0][1] = outgrad[1][0][0] * input[1][0][1] = 0.2 * 6 = 1.2
    // kernel_grad[1][1][1][0] = outgrad[1][0][0] * input[1][1][0] = 0.2 * 7 = 1.4
    // kernel_grad[1][1][1][1] = outgrad[1][0][0] * input[1][1][1] = 0.2 * 8 = 1.6

    let expected_kernel_gradient = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6,
    ];

    // Bias gradients:
    // bias_grad[0] = outgrad[0][0][0] = 0.1
    // bias_grad[1] = outgrad[1][0][0] = 0.2

    let expected_bias_gradient = vec![0.1, 0.2];

    // Verify the results
    for (i, (actual, expected)) in input_gradient
        .iter()
        .zip(expected_input_gradient.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Input gradient mismatch at index {}: {} vs {}",
            i,
            actual,
            expected
        );
    }

    for (i, (actual, expected)) in kernel_gradient
        .iter()
        .zip(expected_kernel_gradient.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Kernel gradient mismatch at index {}: {} vs {}",
            i,
            actual,
            expected
        );
    }

    for (i, (actual, expected)) in bias_gradient
        .iter()
        .zip(expected_bias_gradient.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Bias gradient mismatch at index {}: {} vs {}",
            i,
            actual,
            expected
        );
    }
}
