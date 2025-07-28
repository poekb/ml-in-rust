use cust::{
    memory::{CopyDestination, DeviceBuffer},
    module::Module,
};
use rand::distr::{Distribution, Uniform};

use crate::layers::{Layer, Optimizer, optimizer::OptimizerImpl};

static PTX: &str = include_str!("../kernels/layers/dense_layer.ptx");

/// Simple dense layer implemented using CUDA for GPU acceleration.
pub struct DenseLayer {
    pub weights: DeviceBuffer<f32>,
    pub biases: DeviceBuffer<f32>,
    pub weights_gradient: DeviceBuffer<f32>,
    pub biases_gradient: DeviceBuffer<f32>,
    gradient_count: usize,
    pub input_size: usize,
    pub output_size: usize,
    module: Module,
    weight_optimizer: Box<dyn OptimizerImpl>,
    bias_optimizer: Box<dyn OptimizerImpl>,
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
        optimizer: &Box<dyn Optimizer>,
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
            weight_optimizer: optimizer.instance(input_size * output_size),
            bias_optimizer: optimizer.instance(output_size),
        }
    }

    pub fn boxed(
        input_size: usize,
        output_size: usize,
        initializer: fn(usize, usize, usize) -> Vec<f32>,
        optimizer: &Box<dyn Optimizer>,
    ) -> Box<Self> {
        Box::new(Self::new(input_size, output_size, initializer, optimizer))
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
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.weight_optimizer.optimize(
            &self.weights,
            &self.weights_gradient,
            self.gradient_count,
            stream,
        )?;
        self.bias_optimizer.optimize(
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

    fn serialize_parameters(
        &self,
        writer: &mut dyn std::io::Write,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut host_vec_weights = vec![0.0f32; self.weights.len()];
        let mut host_vec_biases = vec![0.0f32; self.biases.len()];
        self.weights.copy_to(&mut host_vec_weights)?;
        self.biases.copy_to(&mut host_vec_biases)?;

        writer.write_all(&(self.input_size as u64).to_be_bytes())?;
        writer.write_all(&(self.output_size as u64).to_be_bytes())?;

        for &value in &host_vec_weights {
            writer.write_all(&value.to_le_bytes())?;
        }
        for &value in &host_vec_biases {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    fn deserialize_parameters(
        &mut self,
        reader: &mut dyn std::io::Read,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut input_size_bytes = [0u8; 8];
        let mut output_size_bytes = [0u8; 8];
        reader.read_exact(&mut input_size_bytes)?;
        reader.read_exact(&mut output_size_bytes)?;
        let input_size = u64::from_be_bytes(input_size_bytes) as usize;
        let output_size = u64::from_be_bytes(output_size_bytes) as usize;

        if self.input_size != input_size || self.output_size != output_size {
            return Err(Box::from(
                "Input or output size mismatch during deserialization",
            ));
        }

        let weights_count = input_size * output_size;
        let biases_count = output_size;

        let mut host_vec_weights = vec![0.0f32; weights_count];
        for value in &mut host_vec_weights {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
        }

        let mut host_vec_biases = vec![0.0f32; biases_count];
        for value in &mut host_vec_biases {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
        }

        self.weights.copy_from(&host_vec_weights)?;
        self.biases.copy_from(&host_vec_biases)?;

        Ok(())
    }
}
