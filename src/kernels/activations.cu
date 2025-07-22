/// CUDA kernel for ReLU activation forward pass
/// Applies the ReLU function: output = max(0, input)
extern "C" __global__ void relu_forward(
    const float* input,
    float* output,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        if (input[tid] > 0.0f) {
            output[tid] = input[tid];
        }
        else {
            output[tid] = 0.01f * input[tid];
        }
    }
}

/// CUDA kernel for ReLU activation backward pass
/// Computes the gradient: input_gradient = output_gradient * (input > 0)
extern "C" __global__ void relu_backward(
    const float* input,
    const float* output_gradient,
    float* input_gradient,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        input_gradient[tid] = output_gradient[tid] * (input[tid] > 0.0f ? 1.0f : 0.01f);
    }
}


__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/// CUDA kernel for Sigmoid activation forward pass
/// Applies the Sigmoid function: output = 1 / (1 + exp(-input))
extern "C" __global__ void sigmoid_forward(
    const float* input,
    float* output,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        output[tid] = sigmoid(input[tid]);
    }
}

/// CUDA kernel for Sigmoid activation backward pass
/// Computes the gradient: input_gradient = output_gradient * sigmoid(input) * (1 - sigmoid(input))
extern "C" __global__ void sigmoid_backward(
    const float* input,
    const float* output_gradient,
    float* input_gradient,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float sig = sigmoid(input[tid]);
        input_gradient[tid] = output_gradient[tid] * sig * (1.0f - sig);
    }
}
