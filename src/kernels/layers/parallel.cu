// Kernel that duplicates the input buffer to two output buffers.
extern "C" __global__ void duplicate_input(
    const float* input,
    float* output_a,
    float* output_b,
    unsigned int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        output_a[tid] = input[tid];
        output_b[tid] = input[tid];
    }
}

// Kernel that takes in two buffers and concatenates them into a single buffer.
extern "C" __global__ void concat(
    const float* buffer_a,
    const float* buffer_b,
    float* output,
    unsigned int size_a,
    unsigned int size_b
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size_a + size_b) {
        if (tid < size_a) {
            output[tid] = buffer_a[tid];
        }
        else {
            output[tid] = buffer_b[tid - size_a];
        }
    }
}

// Kernel that takes in a single buffer and splits it into two buffers specified by the split sizes.
extern "C" __global__ void split(
    const float* input,
    float* buffer_a,
    float* buffer_b,
    unsigned int size_a,
    unsigned int size_b
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size_a + size_b) {
        if (tid < size_a) {
            buffer_a[tid] = input[tid];
        }
        else {
            buffer_b[tid - size_a] = input[tid];
        }
    }
}

// Kernel for element-wise addition
extern "C" __global__ void element_wise_add(
    const float* input_a,
    const float* input_b,
    float* output,
    unsigned int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        output[tid] = input_a[tid] + input_b[tid];
    }
}

// Kernel for element-wise multiplication
extern "C" __global__ void element_wise_multiply(
    const float* input_a,
    const float* input_b,
    float* output,
    unsigned int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        output[tid] = input_a[tid] * input_b[tid];
    }
}

// Kernel for element-wise max
extern "C" __global__ void element_wise_max(
    const float* input_a,
    const float* input_b,
    float* output,
    unsigned int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        output[tid] = fmaxf(input_a[tid], input_b[tid]);
    }
}

// Kernel for gradient distribution in element-wise multiplication (chain rule)
extern "C" __global__ void multiply_gradient_distribution(
    const float* output_gradient,
    const float* output_a,
    const float* output_b,
    float* gradient_a,
    float* gradient_b,
    unsigned int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        gradient_a[tid] = output_gradient[tid] * output_b[tid];  // dL/da = dL/dz * db
        gradient_b[tid] = output_gradient[tid] * output_a[tid];  // dL/db = dL/dz * da
    }
}

// Kernel for gradient distribution in element-wise max operation
extern "C" __global__ void max_gradient_distribution(
    const float* output_gradient,
    const float* output_a,
    const float* output_b,
    float* gradient_a,
    float* gradient_b,
    unsigned int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        // Gradient flows only to the input that was the maximum
        if (output_a[tid] > output_b[tid]) {
            gradient_a[tid] = output_gradient[tid];
            gradient_b[tid] = 0.0f;
        }
        else {
            gradient_a[tid] = 0.0f;
            gradient_b[tid] = output_gradient[tid];
        }

        // Handle equal case (split gradient equally)
        if (output_a[tid] == output_b[tid]) {
            gradient_a[tid] = output_gradient[tid] * 0.5f;
            gradient_b[tid] = output_gradient[tid] * 0.5f;
        }
    }
}