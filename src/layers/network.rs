use std::vec;

use cust::memory::DeviceBuffer;

use crate::layers::Layer;

pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
    pub input_buffers: Vec<DeviceBuffer<f32>>,
    pub gradient_buffers: Vec<DeviceBuffer<f32>>,
    input_size: usize,
    output_size: usize,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        let mut input_buffers = Vec::new();
        let mut gradient_buffers = Vec::new();

        for (i, layer) in layers.iter().enumerate() {
            if i == layers.len() - 1 {
                break;
            }
            assert_eq!(layer.get_output_size(), layers[i + 1].get_input_size());
            input_buffers
                .push(DeviceBuffer::from_slice(&vec![0.0f32; layer.get_output_size()]).unwrap());
            gradient_buffers
                .push(DeviceBuffer::from_slice(&vec![0.0f32; layer.get_output_size()]).unwrap());
        }
        Self {
            input_size: layers.first().map_or(0, |l| l.get_input_size()),
            output_size: layers.last().map_or(0, |l| l.get_output_size()),
            layers,
            input_buffers,
            gradient_buffers,
        }
    }
    pub fn boxed(layers: Vec<Box<dyn Layer>>) -> Box<Self> {
        Box::new(Self::new(layers))
    }
}

impl Layer for NeuralNetwork {
    fn forward(
        &mut self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let length = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let current_output = if i == length - 1 {
                output
            } else {
                &self.input_buffers[i]
            };
            let current_input = if i == 0 {
                input
            } else {
                &self.input_buffers[i - 1]
            };
            layer.forward(current_input, current_output, stream)?;
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
        let length = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let (current_input_gradient, current_input) = if i == 0 {
                (input_gradient, input)
            } else {
                (&self.gradient_buffers[i - 1], &self.input_buffers[i - 1])
            };

            let current_output_gradient = if i == length - 1 {
                output_gradient
            } else {
                &self.gradient_buffers[i]
            };

            layer.back_propagate(
                current_input_gradient,
                current_input,
                current_output_gradient,
                stream,
            )?;
        }

        Ok(())
    }

    fn optimize(
        &mut self,
        stream: Option<&cust::prelude::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for layer in &mut self.layers {
            layer.optimize(stream)?;
        }
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
        for layer in &self.layers {
            layer.serialize_parameters(writer)?;
        }
        Ok(())
    }

    fn deserialize_parameters(
        &mut self,
        reader: &mut dyn std::io::Read,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for layer in &mut self.layers {
            layer.deserialize_parameters(reader)?;
        }
        Ok(())
    }
}
