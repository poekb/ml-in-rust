use cust::{memory::DeviceBuffer, module::Module};

use crate::layers::{Layer, Optimizer, optimizer::OptimizerImpl};

static PTX: &str = include_str!("../kernels/layers/conv_2d.ptx");

/// Simple dense layer implemented using CUDA for GPU acceleration.
pub struct Conv2DLayer {
    input_depth: usize,
    input_dim_x: usize,
    input_dim_y: usize,

    kernel_dim_x: usize,
    kernel_dim_y: usize,
    kernel_count: usize,
    pub kernel_parameters: DeviceBuffer<f32>,
    pub bias_parameters: DeviceBuffer<f32>,
    pub kernel_gradient: DeviceBuffer<f32>,
    pub bias_gradient: DeviceBuffer<f32>,
    gradient_count: usize,

    input_size: usize,
    output_size: usize,

    module: Module,
    kernel_optimizer: Box<dyn OptimizerImpl>,
    bias_optimizer: Box<dyn OptimizerImpl>,
}

impl Conv2DLayer {
    pub fn new(
        input_depth: usize,
        input_dim_x: usize,
        input_dim_y: usize,
        kernel_dim_x: usize,
        kernel_dim_y: usize,
        kernel_count: usize,
        initializer: fn(usize, usize, usize) -> Vec<f32>,
        optimizer: &Box<dyn Optimizer>,
    ) -> Self {
        let input_size = input_depth * input_dim_x * input_dim_y;
        let output_size =
            kernel_count * (input_dim_x - kernel_dim_x + 1) * (input_dim_y - kernel_dim_y + 1);

        let kernel_parameter_count = kernel_dim_x * kernel_dim_y * input_depth * kernel_count;
        let bias_parameter_count = kernel_count;

        let fan_in = input_depth * kernel_dim_x * kernel_dim_y;
        let fan_out = kernel_count * kernel_dim_x * kernel_dim_y;
        let kernel_parameters =
            DeviceBuffer::from_slice(&initializer(fan_in, fan_out, kernel_parameter_count))
                .expect("Failed to initialize kernel parameters");
        let bias_parameters = DeviceBuffer::from_slice(&vec![0.0f32; bias_parameter_count])
            .expect("Failed to initialize bias parameters");
        let kernel_gradient = DeviceBuffer::from_slice(&vec![0.0f32; kernel_parameter_count])
            .expect("Failed to initialize kernel gradient");
        let bias_gradient = DeviceBuffer::from_slice(&vec![0.0f32; bias_parameter_count])
            .expect("Failed to initialize bias gradient");
        let module = Module::from_ptx(PTX, &[]).expect("Failed to load PTX module");
        Self {
            input_depth,
            input_dim_x,
            input_dim_y,
            kernel_dim_x,
            kernel_dim_y,
            kernel_count,
            kernel_parameters,
            bias_parameters,
            kernel_gradient,
            gradient_count: 0,
            bias_gradient,
            input_size,
            output_size,
            module,
            kernel_optimizer: optimizer.instance(kernel_parameter_count),
            bias_optimizer: optimizer.instance(bias_parameter_count),
        }
    }

    pub fn boxed(
        input_depth: usize,
        input_dim_x: usize,
        input_dim_y: usize,
        kernel_dim_x: usize,
        kernel_dim_y: usize,
        kernel_count: usize,
        initializer: fn(usize, usize, usize) -> Vec<f32>,
        optimizer: &Box<dyn Optimizer>,
    ) -> Box<Self> {
        Box::new(Self::new(
            input_depth,
            input_dim_x,
            input_dim_y,
            kernel_dim_x,
            kernel_dim_y,
            kernel_count,
            initializer,
            optimizer,
        ))
    }
}

impl Layer for Conv2DLayer {
    fn forward(
        &mut self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let block_size = 256;
        let grid_size = ((self.output_size as u32) + block_size - 1) / block_size;
        let forward = self.module.get_function("forward").unwrap();

        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        unsafe {
            cust::launch!(
                forward<<<grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    output.as_device_ptr(),
                    self.kernel_parameters.as_device_ptr(),
                    self.bias_parameters.as_device_ptr(),
                    self.input_depth as u32,
                    self.input_dim_x as u32,
                    self.input_dim_y as u32,
                    self.kernel_dim_x as u32,
                    self.kernel_dim_y as u32,
                    self.kernel_count as u32,
                    (self.input_dim_x - self.kernel_dim_x + 1) as u32,
                    (self.input_dim_y - self.kernel_dim_y + 1) as u32,
                )
            )?;
        }
        if stream.is_none() {
            real_stream.synchronize()?;
        }

        Ok(())
    }

    fn back_propagate(
        &mut self,
        input_gradient: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        output_gradient: &DeviceBuffer<f32>,
        stream: Option<&cust::prelude::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let block_size = 256;
        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        let output_dim_x = (self.input_dim_x - self.kernel_dim_x + 1) as u32;
        let output_dim_y = (self.input_dim_y - self.kernel_dim_y + 1) as u32;

        // 1. Compute input gradient (dL/dX)
        let input_grid_size = (self.input_size as u32 + block_size - 1) / block_size;
        let backward_input_fn = self.module.get_function("backward_input")?;
        unsafe {
            cust::launch!(
                backward_input_fn<<<input_grid_size, block_size, 0, real_stream>>>(
                    output_gradient.as_device_ptr(),
                    input_gradient.as_device_ptr(),
                    self.kernel_parameters.as_device_ptr(),
                    self.input_depth as u32,
                    self.input_dim_x as u32,
                    self.input_dim_y as u32,
                    self.kernel_dim_x as u32,
                    self.kernel_dim_y as u32,
                    self.kernel_count as u32,
                    output_dim_x,
                    output_dim_y
                )
            )?;
        }

        // 2. Compute kernel gradient (dL/dW)
        let kernel_grid_size = (self.kernel_parameters.len() as u32 + block_size - 1) / block_size;

        let backward_kernel_fn = self.module.get_function("backward_kernel")?;
        unsafe {
            cust::launch!(
                backward_kernel_fn<<<kernel_grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    output_gradient.as_device_ptr(),
                    self.kernel_gradient.as_device_ptr(),
                    self.input_depth as u32,
                    self.input_dim_x as u32,
                    self.input_dim_y as u32,
                    self.kernel_dim_x as u32,
                    self.kernel_dim_y as u32,
                    self.kernel_count as u32,
                    output_dim_x,
                    output_dim_y
                )
            )?;
        }

        // 3. Compute bias gradient (dL/dB)
        // Launch one thread per filter/channel.
        let bias_grid_size = (self.kernel_count as u32 + block_size - 1) / block_size;
        let backward_bias_fn = self.module.get_function("backward_bias")?;
        unsafe {
            cust::launch!(
                backward_bias_fn<<<bias_grid_size, block_size, 0, real_stream>>>(
                    output_gradient.as_device_ptr(),
                    self.bias_gradient.as_device_ptr(),
                    self.kernel_count as u32,
                    output_dim_x,
                    output_dim_y
                )
            )?;
        }

        if stream.is_none() {
            real_stream.synchronize()?;
        }
        self.gradient_count += 1;
        Ok(())
    }

    fn optimize(
        &mut self,
        stream: Option<&cust::prelude::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.kernel_optimizer.optimize(
            &self.kernel_parameters,
            &self.kernel_gradient,
            self.gradient_count,
            stream,
        )?;
        self.bias_optimizer.optimize(
            &self.bias_parameters,
            &self.bias_gradient,
            self.gradient_count,
            stream,
        )?;
        Ok(())
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn get_output_size(&self) -> usize {
        self.output_size
    }
}
