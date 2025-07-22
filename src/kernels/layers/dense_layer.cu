extern "C" __global__ void forward(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int input_size,
    int output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + tid];
        }
        output[tid] = sum + biases[tid];
    }
}

extern "C" __global__ void backward(
    const float* input,
    const float* weights,
    const float* biases,
    const float* output_gradient,
    float* input_gradient,
    float* weights_gradient,
    float* biases_gradient,
    int input_size,
    int output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input_size) {
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            sum += output_gradient[i] * weights[tid * output_size + i];
        }
        input_gradient[tid] = sum;
    }
    if (tid < output_size) {
        biases_gradient[tid] += output_gradient[tid];
        for (int i = 0; i < input_size; i++) {
            weights_gradient[i * output_size + tid] += input[i] * output_gradient[tid];
        }
    }
}
