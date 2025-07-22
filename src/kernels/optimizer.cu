extern "C" __global__ void optimize(
    float* parameters,
    float* gradient,
    unsigned int gradient_count,
    unsigned int count,
    float learning_rate
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        parameters[tid] -= learning_rate * gradient[tid] / gradient_count;
        gradient[tid] = 0.0f; // Reset gradient after optimization
    }
}