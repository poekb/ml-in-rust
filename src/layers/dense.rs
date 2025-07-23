use cust::{memory::DeviceBuffer, module::Module};
use rand::distr::{Distribution, Uniform};

use crate::layers::{Layer, Optimizer};

static PTX: &str = include_str!("../kernels/layers/dense_layer.ptx");

/// Simple dense layer implemented using CUDA for GPU acceleration.
pub struct DenseLayer {
    pub weights: DeviceBuffer<f32>,
    pub biases: DeviceBuffer<f32>,
    weights_gradient: DeviceBuffer<f32>,
    biases_gradient: DeviceBuffer<f32>,
    gradient_count: usize,
    pub input_size: usize,
    pub output_size: usize,
    module: Module,
}

pub fn xavier_initializer(input_size: usize, output_size: usize, count: usize) -> Vec<f32> {
    let limit = (6.0 / (input_size + output_size) as f32).sqrt();
    let dist = Uniform::new(-limit, limit).unwrap();
    let mut rng = rand::rng();
    (0..count).map(|_| dist.sample(&mut rng)).collect()
}

pub fn he_initializer(input_size: usize, output_size: usize, count: usize) -> Vec<f32> {
    let limit = (2.0 / (input_size + output_size) as f32).sqrt();
    let dist = Uniform::new(-limit, limit).unwrap();
    let mut rng = rand::rng();
    (0..count).map(|_| dist.sample(&mut rng)).collect()
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        initializer: fn(usize, usize, usize) -> Vec<f32>,
    ) -> Self {
        let weights = DeviceBuffer::from_slice(&initializer(
            input_size,
            output_size,
            input_size * output_size,
        ))
        .unwrap();
        let biases = DeviceBuffer::from_slice(&vec![0.0f32; output_size]).unwrap();
        let weights_gradient =
            DeviceBuffer::from_slice(&vec![0.0f32; input_size * output_size]).unwrap();
        let biases_gradient = DeviceBuffer::from_slice(&vec![0.0f32; output_size]).unwrap();
        let module = Module::from_ptx(PTX, &[]).expect("Failed to load PTX module");
        Self {
            weights,
            biases,
            weights_gradient,
            biases_gradient,
            gradient_count: 0,
            input_size,
            output_size,
            module,
        }
    }

    pub fn boxed(
        input_size: usize,
        output_size: usize,
        initializer: fn(usize, usize, usize) -> Vec<f32>,
    ) -> Box<Self> {
        Box::new(Self::new(input_size, output_size, initializer))
    }
}

impl Layer for DenseLayer {
    fn forward(
        &mut self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let block_size = 256;
        let grid_size = ((self.output_size as u32) + block_size - 1) / block_size;
        let forward = self.module.get_function("forward")?;

        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        unsafe {
            cust::launch!(
                forward<<<grid_size, block_size,0, real_stream>>>(
                    input.as_device_ptr(),
                    self.weights.as_device_ptr(),
                    self.biases.as_device_ptr(),
                    output.as_device_ptr(),
                    self.input_size as u32,
                    self.output_size as u32
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
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let block_size = 256;

        let grid_size =
            ((usize::max(self.input_size, self.output_size) as u32) + block_size - 1) / block_size;
        let module = Module::from_ptx(PTX, &[])?;
        let backward = module.get_function("backward")?;
        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };
        unsafe {
            cust::launch!(
                backward<<<grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    self.weights.as_device_ptr(),
                    self.biases.as_device_ptr(),
                    output_gradient.as_device_ptr(),

                    input_gradient.as_device_ptr(),
                    self.weights_gradient.as_device_ptr(),
                    self.biases_gradient.as_device_ptr(),

                    self.input_size as u32,
                    self.output_size as u32
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
        optimizer: &Box<dyn Optimizer>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        optimizer.optimize(
            &self.weights,
            &self.weights_gradient,
            self.gradient_count,
            stream,
        )?;
        optimizer.optimize(
            &self.biases,
            &self.biases_gradient,
            self.gradient_count,
            stream,
        )?;
        self.gradient_count = 0;

        Ok(())
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn get_output_size(&self) -> usize {
        self.output_size
    }
}
