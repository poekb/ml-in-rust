# GPU-Accelerated Machine Learning in Rust

This project is a simple neural network library built from scratch in Rust, with core computations accelerated on an NVIDIA GPU using custom CUDA kernels. It demonstrates how to build, train, and evaluate a neural network on the classic MNIST dataset of handwritten digits.

> [!WARNING]
> **Disclaimer:** This project is in a very early stage of development and should be considered a proof of concept. It is intended to demonstrate the integration of Rust and CUDA for machine learning, not as a production-ready library. This demo was put together in two afternoons, so it is not optimized for performance or usability. The code is intended to be a starting point for further development and experimentation.

## Features

-   **Custom Neural Network:** A modular, layer-based neural network implementation in pure Rust.
-   **GPU Acceleration:** Forward and backward propagation for dense and activation layers are executed on the GPU using custom CUDA C++ kernels.
-   **Rust + CUDA:** Demonstrates the integration of Rust with CUDA for high-performance computing using the `cust` crate.
-   **Build-Time Kernel Compilation:** A `build.rs` script automatically compiles CUDA (`.cu`) files into PTX (Parallel Thread Execution) assembly, which is then loaded at runtime.
-   **Included Components:**
    -   Dense (fully connected) layers.
    -   ReLU and Sigmoid activation functions.
    -   Xavier weight initialization.
    -   Stochastic Gradient Descent (SGD) optimizer.

## Prerequisites

-   **Rust Toolchain:** Install from [rustup.rs](https://rustup.rs/).
-   **NVIDIA GPU:** An NVIDIA GPU with CUDA support is required.
-   **CUDA Toolkit:** The NVIDIA CUDA Toolkit must be installed. The build script relies on the `nvcc` compiler and the `CUDA_HOME` environment variable. You can download it from the [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads) website.
-   **MNIST Dataset:** The dataset is included in the git repo for demonstrating just how fast a network can be trained for >97% accuracy

    You can get the files from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd gpu-accelerated-machine-learning
    ```

2.  **Build and run the project:**
    For the best performance, run in release mode.
    ```bash
    cargo run --release
    ```

The program will build the CUDA kernels, create the neural network, train it on the MNIST training set for 15 epochs, and finally report the model's accuracy on the test set.

## Future Features
-   Support for serialization and deserialization of the model.
-   More advanced optimizers (e.g., Adam, RMSprop).
-   Support for dropout and batch normalization layers.
-   Support for more complex architectures (e.g., convolutional layers).

## Project Structure

```
.
├── MNIST/                  # MNIST dataset files (must be added manually)
├── kernels/
│   ├── layers/
│   │   └── dense_layer.cu  # CUDA kernel for the Dense layer
│   └── activations.cu      # CUDA kernels for activation functions
├── src/
│   ├── layers/             # Rust implementation of network layers
│   │   ├── activation.rs
│   │   ├── dense.rs
│   │   └── ...
│   ├── mnist_loader.rs     # Loader for the MNIST dataset
│   ├── lib.rs              # Library root
│   └── main.rs             # Main executable to build and run the network
├── build.rs                # Build script to compile CUDA kernels
└── Cargo.toml
```

-   **`kernels/`**: Contains all CUDA C++ (`.cu`) source files. These are compiled to PTX format by the `build.rs` script.
-   **`src/`**: Contains all the Rust source code.
    -   **`src/main.rs`**: Defines the network architecture, loads the data, and runs the training and evaluation loop.
    -   **`src/layers/`**: Defines the building blocks of the network, such as `DenseLayer` and `ActivationLayer`. These modules are responsible for calling the appropriate CUDA kernels.
-   **`build.rs`**: This script is executed by Cargo before building the crate. It finds all `.cu` files in the `kernels` directory and uses `nvcc` (the CUDA compiler) to compile them into PTX files, which are included in the final binary.