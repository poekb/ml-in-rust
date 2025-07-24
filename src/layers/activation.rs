use cust::{memory::DeviceBuffer, module::Module};

use crate::layers::Layer;

static PTX: &str = include_str!("../kernels/activations.ptx");

pub const RELU: ActivationFunction = ActivationFunction {
    forward: "relu_forward",
    backward: "relu_backward",
};

pub const SIGMOID: ActivationFunction = ActivationFunction {
    forward: "sigmoid_forward",
    backward: "sigmoid_backward",
};

pub const SOFTMAX: ActivationFunction = ActivationFunction {
    forward: "softmax_forward",
    backward: "softmax_backward",
};

pub struct ActivationFunction {
    forward: &'static str,
    backward: &'static str,
}

pub struct ActivationLayer {
    module: Module,
    function: ActivationFunction,
    size: usize,
}

impl ActivationLayer {
    pub fn new(function: ActivationFunction, size: usize) -> Self {
        let module = Module::from_ptx(PTX, &[]).expect("Failed to load PTX module");
        Self {
            module,
            function,
            size,
        }
    }
    pub fn boxed(function: ActivationFunction, size: usize) -> Box<Self> {
        Box::new(Self::new(function, size))
    }
}

impl Layer for ActivationLayer {
    fn forward(
        &mut self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let block_size = 256;
        let grid_size = ((input.len() as u32) + block_size - 1) / block_size;
        let forward = self.module.get_function(self.function.forward).unwrap();

        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        unsafe {
            cust::launch!(
                forward<<<grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    output.as_device_ptr(),
                    input.len() as u32
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
        let grid_size = ((input.len() as u32) + block_size - 1) / block_size;
        let backward = self.module.get_function(self.function.backward).unwrap();
        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };
        unsafe {
            cust::launch!(
                backward<<<grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    output_gradient.as_device_ptr(),
                    input_gradient.as_device_ptr(),
                    input.len() as u32
                )
            )?;
        }
        if stream.is_none() {
            real_stream.synchronize()?;
        }
        Ok(())
    }
    fn optimize(
        &mut self,
        _stream: Option<&cust::prelude::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // ReLU does not have parameters to optimize, so this is a no-op
        Ok(())
    }

    fn get_input_size(&self) -> usize {
        self.size
    }

    fn get_output_size(&self) -> usize {
        self.size
    }
}
