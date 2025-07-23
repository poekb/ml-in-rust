static PTX: &str = include_str!("../kernels/layers/max_pool_2d.ptx");

use crate::layers::{Layer, Optimizer};
use cust::{memory::DeviceBuffer, module::Module};
use std::{error::Error, vec};
pub struct MaxPool2DLayer {
    pub depth: usize,
    pub input_dim_x: usize,
    pub input_dim_y: usize,

    pub pool_size: usize,
    pub stride: usize,
    pub output_dim_x: usize,
    pub output_dim_y: usize,
    selected_indices: DeviceBuffer<u32>, // Buffer to store indices of max values on each output position
    module: Module,
}

impl MaxPool2DLayer {
    pub fn new(
        depth: usize,
        input_dim_x: usize,
        input_dim_y: usize,
        pool_size: usize,
        stride: usize,
    ) -> Self {
        let output_dim_x = (input_dim_x - pool_size) / stride + 1;
        let output_dim_y = (input_dim_y - pool_size) / stride + 1;

        let selected_indices =
            DeviceBuffer::from_slice(&vec![0u32; output_dim_x * output_dim_y * depth]).unwrap();
        let module = Module::from_ptx(PTX, &[]).expect("Failed to load PTX module");

        Self {
            depth,
            input_dim_x,
            input_dim_y,
            pool_size,
            stride,
            output_dim_x,
            output_dim_y,
            selected_indices,
            module,
        }
    }
    pub fn boxed(
        depth: usize,
        input_dim_x: usize,
        input_dim_y: usize,
        pool_size: usize,
        stride: usize,
    ) -> Box<Self> {
        Box::new(Self::new(
            depth,
            input_dim_x,
            input_dim_y,
            pool_size,
            stride,
        ))
    }
}

impl Layer for MaxPool2DLayer {
    fn forward(
        &mut self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn Error>> {
        let block_size = 256;
        let output_size = self.output_dim_x * self.output_dim_y * self.depth;
        let grid_size = (output_size as u32 + block_size - 1) / block_size;

        let forward_fn = self.module.get_function("forward")?;

        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        unsafe {
            cust::launch!(
                forward_fn<<<grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    output.as_device_ptr(),
                    self.selected_indices.as_device_ptr(),
                    self.depth as u32,
                    self.input_dim_x as u32,
                    self.input_dim_y as u32,
                    self.pool_size as u32,
                    self.stride as u32,
                    self.output_dim_x as u32,
                    self.output_dim_y as u32
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
        _input: &DeviceBuffer<f32>, // Not needed with this strategy
        output_gradient: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn Error>> {
        // Note: The input_gradient buffer must be zeroed out before this call.
        let block_size = 256;
        let output_size = self.output_dim_x * self.output_dim_y * self.depth;
        let grid_size = (output_size as u32 + block_size - 1) / block_size;

        let backward_fn = self.module.get_function("backward")?;

        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        // Reset the gradient buffer to zero before backpropagation
        let reset_fn = self.module.get_function("reset_gradient")?;

        unsafe {
            cust::launch!(
                reset_fn<<<(input_gradient.len() as u32 + block_size - 1) / block_size, block_size, 0, real_stream>>>(
                    input_gradient.as_device_ptr(),
                    input_gradient.len() as u32
                )
            )?;
            cust::launch!(
                backward_fn<<<grid_size, block_size, 0, real_stream>>>(
                    output_gradient.as_device_ptr(),
                    input_gradient.as_device_ptr(),
                    self.selected_indices.as_device_ptr(),
                    output_size as u32
                )
            )?;
        }

        if stream.is_none() {
            real_stream.synchronize()?;
        }
        Ok(())
    }

    // MaxPool has no learnable parameters, so optimize is a no-op.
    fn optimize(
        &mut self,
        _optimizer: &Box<dyn Optimizer>,
        _stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn get_input_size(&self) -> usize {
        self.depth * self.input_dim_x * self.input_dim_y
    }
    fn get_output_size(&self) -> usize {
        self.depth * self.output_dim_x * self.output_dim_y
    }
}
