// input.as_device_ptr(),
// output.as_device_ptr(),
// self.kernel_parameters.as_device_ptr(),
// self.bias_parameters.as_device_ptr(),
// self.input_depth as u32,
// self.input_dim_x as u32,
// self.input_dim_y as u32,
// self.kernel_dim_x as u32,
// self.kernel_dim_y as u32,
// self.kernel_count as u32,
// (self.input_dim_x - self.kernel_dim_x + 1) as u32,
// (self.input_dim_y - self.kernel_dim_y + 1) as u32,
extern "C" __global__ void forward(
    const float* input,
    float* output,
    const float* kernel,
    const float* bias,
    unsigned int input_depth,
    unsigned int input_dim_x,
    unsigned int input_dim_y,
    unsigned int kernel_dim_x,
    unsigned int kernel_dim_y,
    unsigned int kernel_count,
    unsigned int output_dim_x,
    unsigned int output_dim_y
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_dim_x * output_dim_y * kernel_count) {
        int i = tid / (output_dim_x * output_dim_y);
        int oy = (tid % (output_dim_x * output_dim_y)) / output_dim_x;
        int ox = tid % output_dim_x;

        float sum = 0.0f;
        for (int j = 0; j < input_depth; j++) {
            for (int ky = 0; ky < kernel_dim_y; ky++) {
                for (int kx = 0; kx < kernel_dim_x; kx++) {
                    int input_x = ox + kx;
                    int input_y = oy + ky;

                    sum += input[j * (input_dim_x * input_dim_y)
                        + input_y * input_dim_x
                        + input_x]
                        *
                        kernel[i * (input_depth * kernel_dim_y * kernel_dim_x)
                        + j * (kernel_dim_y * kernel_dim_x)
                        + ky * kernel_dim_x
                        + kx];
                }
            }
        }
        output[tid] = sum + bias[i];
    }
}


// Kernel to compute the input gradient (dL/dX)
// Launched with one thread per input element.
extern "C" __global__ void backward_input(
    const float* output_gradient,
    float* input_gradient,
    const float* kernel,
    unsigned int input_depth,
    unsigned int input_dim_x,
    unsigned int input_dim_y,
    unsigned int kernel_dim_x,
    unsigned int kernel_dim_y,
    unsigned int kernel_count,
    unsigned int output_dim_x,
    unsigned int output_dim_y
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int input_size = input_depth * input_dim_x * input_dim_y;

    if (tid < input_size) {
        int j = tid / (input_dim_x * input_dim_y); // Input channel
        int iy = (tid % (input_dim_x * input_dim_y)) / input_dim_x; // Input y position
        int ix = tid % input_dim_x; // Input x position

        float sum = 0.0f;

        for (int i = 0; i < kernel_count; i++) { // Output channel
            for (int ky = 0; ky < kernel_dim_y; ky++) {
                for (int kx = 0; kx < kernel_dim_x; kx++) {
                    // For backward pass, we need to flip the kernel indices
                    int flipped_ky = kernel_dim_y - 1 - ky;
                    int flipped_kx = kernel_dim_x - 1 - kx;

                    // Calculate output position that contributed to this input
                    int oy = iy - flipped_ky;
                    int ox = ix - flipped_kx;

                    // Check if output position is valid
                    if (oy >= 0 && oy < output_dim_y && ox >= 0 && ox < output_dim_x) {
                        sum += output_gradient[i * (output_dim_x * output_dim_y) + oy * output_dim_x + ox]
                            *
                            kernel[
                                i * (input_depth * kernel_dim_y * kernel_dim_x) +
                                    j * (kernel_dim_y * kernel_dim_x) +
                                    flipped_ky * kernel_dim_x +
                                    flipped_kx
                            ];
                    }
                }
            }
        }

        input_gradient[tid] = sum;
    }
}

// Kernel to compute the kernel/weight gradient (dL/dW)
// Launched with one thread per output element.
// input.as_device_ptr(),
// output_gradient.as_device_ptr(),
// self.kernel_gradient.as_device_ptr(),
// self.input_depth as u32,
// self.input_dim_x as u32,
// self.input_dim_y as u32,
// self.kernel_dim_x as u32,
// self.kernel_dim_y as u32,
// self.kernel_count as u32,
// output_dim_x,
// output_dim_y
extern "C" __global__ void backward_kernel(
    const float* input,
    const float* output_gradient,
    float* kernel_gradient,
    unsigned int input_depth,
    unsigned int input_dim_x,
    unsigned int input_dim_y,
    unsigned int kernel_dim_x,
    unsigned int kernel_dim_y,
    unsigned int kernel_count,
    unsigned int output_dim_x,
    unsigned int output_dim_y
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int kernel_size = kernel_count * input_depth * kernel_dim_x * kernel_dim_y;

    if (tid < kernel_size) {
        int i = tid / (kernel_dim_x * kernel_dim_y * input_depth);
        int j = (tid / (kernel_dim_x * kernel_dim_y)) % input_depth;

        int ky = (tid % (kernel_dim_x * kernel_dim_y)) / kernel_dim_x;
        int kx = (tid % kernel_dim_x);
        float sum = 0.0;
        for (int oy = 0; oy < output_dim_y; oy++) {
            for (int ox = 0; ox < output_dim_x; ox++) {
                int iy = oy + ky;
                int ix = ox + kx;

                sum += input[
                    j * (input_dim_x * input_dim_y)
                        + iy * input_dim_x
                        + ix]
                    * output_gradient[
                        i * (output_dim_x * output_dim_y)
                            + oy * output_dim_x
                            + ox];
            }
        }

        kernel_gradient[tid] += sum;
    }
}

// Kernel to compute the bias gradient (dL/dB)
// Launched with one thread per filter/channel.
extern "C" __global__ void backward_bias(
    const float* output_gradient,
    float* bias_gradient,
    unsigned int kernel_count,
    unsigned int output_dim_x,
    unsigned int output_dim_y
) {
    // Each thread calculates the gradient for one bias value.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < kernel_count) {
        float sum = 0.0f;
        unsigned int output_channel_size = output_dim_x * output_dim_y;

        // Sum all the output gradients for the current channel 'i'.
        for (int j = 0; j < output_channel_size; j++) {
            sum += output_gradient[i * output_channel_size + j];
        }

        // Accumulate the gradient for the batch.
        // This is safe because each thread 'i' writes to a unique bias_gradient[i].
        bias_gradient[i] += sum;
    }
}