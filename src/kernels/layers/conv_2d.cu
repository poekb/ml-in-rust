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
        output[tid] = sum + bias[tid];
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
        // Decompose thread ID into input coordinates (, iy, ix)
        int j = tid / (input_dim_x * input_dim_y);
        int iy = (tid % (input_dim_x * input_dim_y)) / input_dim_x;
        int ix = tid % input_dim_x;

        float sum = 0.0;

        for (int i = 0; i < kernel_count; i++) {
            for (int ky = 0; ky < kernel_dim_y; ky++) {
                for (int kx = 0; kx < kernel_dim_x; kx++) {
                    int ox = ix + kx - kernel_dim_x + 1;
                    int oy = iy + ky - kernel_dim_y + 1;
                    if (0 <= ox && ox < output_dim_x && 0 <= oy && oy < output_dim_y) {
                        sum += output_gradient[i * (kernel_dim_x * kernel_dim_y) + oy * kernel_dim_x + ox]
                            * kernel[i * (input_depth * kernel_dim_y * kernel_dim_x) + j * (kernel_dim_x * kernel_dim_y)
                            // Because this is a convolution and not a cross-correlation we flip the kernel
                            + (kernel_dim_y - ky - 1) * kernel_dim_x + (kernel_dim_x - kx - 1)
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

                sum += input[
                    j * (input_dim_x * input_dim_y)
                        + oy * input_dim_x
                        + ox]
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
// Launched with one thread per output element.
// output_gradient.as_device_ptr(),
// self.bias_gradient.as_device_ptr(),
// self.output_size as u32
extern "C" __global__ void backward_bias(
    const float* output_gradient,
    float* bias_gradient,
    unsigned int output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_size) {
        bias_gradient[tid] += output_gradient[tid];
    }
}