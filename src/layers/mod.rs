use cust::{
    memory::{CopyDestination, DeviceBuffer},
    stream::Stream,
};

pub mod activation;
pub mod conv_2d;
pub mod dense;
pub mod max_pool_2d;
pub mod network;
pub mod optimizer;

use optimizer::Optimizer;

pub trait Layer {
    fn forward(
        &mut self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: Option<&Stream>,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn back_propagate(
        &mut self,
        input_gradient: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        output_gradient: &DeviceBuffer<f32>,
        stream: Option<&Stream>,
    ) -> Result<(), Box<dyn std::error::Error>>;

    fn optimize(&mut self, stream: Option<&Stream>) -> Result<(), Box<dyn std::error::Error>>;

    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
}

pub struct LayerWrapper {
    input_buffer: DeviceBuffer<f32>,
    output_buffer: DeviceBuffer<f32>,
    input_gradient_buffer: DeviceBuffer<f32>,
    output_gradient_buffer: DeviceBuffer<f32>,
    layer: Box<dyn Layer>,
}

impl LayerWrapper {
    pub fn new(layer: Box<dyn Layer>) -> Result<Self, Box<dyn std::error::Error>> {
        let input_size = layer.get_input_size();
        let output_size = layer.get_output_size();
        let input_buffer = DeviceBuffer::from_slice(&vec![0.0f32; input_size])?;
        let output_buffer = DeviceBuffer::from_slice(&vec![0.0f32; output_size])?;
        let input_gradient_buffer = DeviceBuffer::from_slice(&vec![0.0f32; input_size])?;
        let output_gradient_buffer = DeviceBuffer::from_slice(&vec![0.0f32; output_size])?;
        Ok(Self {
            input_buffer,
            output_buffer,
            input_gradient_buffer,
            output_gradient_buffer,
            layer,
        })
    }

    pub fn infer(&mut self, input: &Vec<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.input_buffer.copy_from(input)?;
        let stream = Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;
        self.layer
            .forward(&self.input_buffer, &self.output_buffer, Some(&stream))?;
        stream.synchronize()?;
        let mut result = vec![0.0f32; self.layer.get_output_size()];
        self.output_buffer.copy_to(result.as_mut_slice())?;
        result = result.iter().map(|&x| x.exp()).collect::<Vec<f32>>();
        let sum: f32 = result.iter().sum();
        result = result.iter().map(|&x| x / sum).collect::<Vec<f32>>();
        Ok(result)
    }

    pub fn back_propagate(&mut self, target: &Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let mut output = vec![0.0f32; self.layer.get_output_size()];
        self.output_buffer.copy_to(output.as_mut_slice())?;

        // Apply softmax with numerical stability
        let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let output: Vec<f32> = output.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = output.iter().sum();
        let output: Vec<f32> = output.iter().map(|&x| x / (sum.max(1e-10))).collect();

        let output_gradient = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o - t)
            .collect::<Vec<f32>>();
        self.output_gradient_buffer.copy_from(&output_gradient)?;
        let stream = Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;
        self.layer.back_propagate(
            &self.input_gradient_buffer,
            &self.input_buffer,
            &self.output_gradient_buffer,
            Some(&stream),
        )?;
        stream.synchronize()?;
        Ok(())
    }

    pub fn optimize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stream = Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;
        self.layer.optimize(Some(&stream))?;
        stream.synchronize()?;
        Ok(())
    }
}
