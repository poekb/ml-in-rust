extern "C" __global__ void adam_optimize(
    float* parameters,
    float* gradient,
    float* m, // First moment vector
    float* v, // Second moment vector
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    unsigned int gradient_count,
    unsigned int count,
    unsigned int t // Timestep
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        // Calculate the average gradient for the current parameter
        float g = gradient[tid] / (float)gradient_count;

        // Update biased first moment estimate
        m[tid] = beta1 * m[tid] + (1.0f - beta1) * g;

        // Update biased second raw moment estimate
        v[tid] = beta2 * v[tid] + (1.0f - beta2) * (g * g);

        // Compute bias-corrected first moment estimate
        // The timestep t is passed from the host and is the same for all threads.
        float m_hat = m[tid] / (1.0f - powf(beta1, t));

        // Compute bias-corrected second raw moment estimate
        float v_hat = v[tid] / (1.0f - powf(beta2, t));

        // Update parameters
        parameters[tid] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);

        // Reset gradient for the next batch
        gradient[tid] = 0.0f;
    }
}

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