// Kernel that takes in a single buffer and splits it into two buffers specified by the split sizes.
// Launched with one thread per output element.
extern "C" __global__ void split(
    const float* input,
    float* output_a,
    float* output_b,
    unsigned int input_size_a,
    unsigned int input_size_b
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size_a + input_size_b) {
        if (tid < input_size_a) {
            output_a[tid] = input[tid];
        }
        else {
            output_b[tid - input_size_a] = input[tid];
        }
    }
}

// Kernel that takes in two buffers and concatenates them into a single buffer.
// Launched with one thread per input element.
extern "C" __global__ void concat(
    const float* input_a,
    const float* input_b,
    float* output,
    unsigned int input_size_a,
    unsigned int input_size_b
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size_a + input_size_b) {
        if (tid < input_size_a) {
            output[tid] = input_a[tid];
        }
        else {
            output[tid] = input_b[tid - input_size_a];
        }
    }
}