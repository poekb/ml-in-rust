extern "C" __global__ void forward(
    const float* input,
    float* output,
    unsigned int* selected_indices, // Store indices of max values
    unsigned int input_depth,
    unsigned int input_dim_x,
    unsigned int input_dim_y,
    unsigned int pool_size,
    unsigned int stride,
    unsigned int output_dim_x,
    unsigned int output_dim_y
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = output_dim_x * output_dim_y * input_depth;

    if (tid < output_size) {
        int d = tid / (output_dim_x * output_dim_y);
        int oy = (tid % (output_dim_x * output_dim_y)) / output_dim_x;
        int ox = tid % output_dim_x;

        float max_val = -INFINITY;
        int max_idx = -1;

        int start_x = ox * stride;
        int start_y = oy * stride;

        for (int py = 0; py < pool_size; py++) {
            for (int px = 0; px < pool_size; px++) {
                int ix = start_x + px;
                int iy = start_y + py;

                if (ix < input_dim_x && iy < input_dim_y) {
                    int input_idx = d * (input_dim_x * input_dim_y) + iy * input_dim_x + ix;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }
        output[tid] = max_val;
        selected_indices[tid] = max_idx;
    }
}

extern "C" __global__ void reset_gradient(
    float* input_gradient,
    unsigned int input_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        input_gradient[tid] = 0.0f; // Reset gradient to zero
    }
}

extern "C" __global__ void backward(
    const float* output_gradient,
    float* input_gradient,
    const unsigned int* selected_indices, // Use stored indices
    unsigned int output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_size) {
        // Get the index of the input that was the max
        int input_idx = selected_indices[tid];

        // *** FIX: Check for a valid index before adding the gradient ***
        if (input_idx != -1) {
            // Atomically add the gradient. This is needed in case pools overlap (stride < pool_size)
            // and multiple outputs map to the same input.
            atomicAdd(&input_gradient[input_idx], output_gradient[tid]);
        }
    }
}