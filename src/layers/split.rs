// Implement a layer that splits the input buffer into two parts, and executes a boxed layer on each part then merges the outputs back.

use cust::memory::DeviceBuffer;

use crate::layers::Layer;

static PTX: &str = include_str!("../kernels/layers/split.ptx");

pub struct SplitLayer {
    input_size_a: usize,
    input_size_b: usize,
    input_a: DeviceBuffer<f32>,
    input_b: DeviceBuffer<f32>,
    input_gradient_a: DeviceBuffer<f32>,
    input_gradient_b: DeviceBuffer<f32>,
    output_size_a: usize,
    output_size_b: usize,
    output_a: DeviceBuffer<f32>,
    output_b: DeviceBuffer<f32>,
    output_gradient_a: DeviceBuffer<f32>,
    output_gradient_b: DeviceBuffer<f32>,
    layer_a: Box<dyn Layer>,
    layer_b: Box<dyn Layer>,
    module: cust::module::Module,
}

impl SplitLayer {
    pub fn new(
        layer_a: Box<dyn Layer>,
        layer_b: Box<dyn Layer>,
        input_size_a: usize,
        input_size_b: usize,
    ) -> Self {
        let output_size_a = layer_a.get_output_size();
        let output_size_b = layer_b.get_output_size();

        let input_a = DeviceBuffer::from_slice(&vec![0.0f32; input_size_a]).unwrap();
        let input_b = DeviceBuffer::from_slice(&vec![0.0f32; input_size_b]).unwrap();
        let input_gradient_a = DeviceBuffer::from_slice(&vec![0.0f32; input_size_a]).unwrap();
        let input_gradient_b = DeviceBuffer::from_slice(&vec![0.0f32; input_size_b]).unwrap();
        let output_a = DeviceBuffer::from_slice(&vec![0.0f32; output_size_a]).unwrap();
        let output_b = DeviceBuffer::from_slice(&vec![0.0f32; output_size_b]).unwrap();
        let output_gradient_a = DeviceBuffer::from_slice(&vec![0.0f32; output_size_a]).unwrap();
        let output_gradient_b = DeviceBuffer::from_slice(&vec![0.0f32; output_size_b]).unwrap();

        let module = cust::module::Module::from_ptx(PTX, &[]).expect("Failed to load PTX module");

        Self {
            input_size_a,
            input_size_b,
            input_a,
            input_b,
            input_gradient_a,
            input_gradient_b,
            output_size_a,
            output_size_b,
            output_a,
            output_b,
            output_gradient_a,
            output_gradient_b,
            layer_a,
            layer_b,
            module,
        }
    }

    pub fn boxed(
        layer_a: Box<dyn Layer>,
        layer_b: Box<dyn Layer>,
        input_size_a: usize,
        input_size_b: usize,
    ) -> Box<Self> {
        Box::new(Self::new(layer_a, layer_b, input_size_a, input_size_b))
    }
}

impl Layer for SplitLayer {
    fn forward(
        &mut self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        let split_fn = self.module.get_function("split")?;

        let block_size = 256;
        let grid_size =
            ((self.input_size_a + self.input_size_b) as u32 + block_size - 1) / block_size;

        unsafe {
            cust::launch!(
                split_fn<<<grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    self.input_a.as_device_ptr(),
                    self.input_b.as_device_ptr(),
                    self.input_size_a as u32,
                    self.input_size_b as u32
                )
            )?;
        }

        self.layer_a
            .forward(&self.input_a, &self.output_a, stream)?;
        self.layer_b
            .forward(&self.input_b, &self.output_b, stream)?;

        let concat_fn = self.module.get_function("concat")?;

        let out_grid_size =
            ((self.output_size_a + self.output_size_b) as u32 + block_size - 1) / block_size;

        unsafe {
            cust::launch!(
                concat_fn<<<out_grid_size, block_size, 0, real_stream>>>(
                    self.output_a.as_device_ptr(),
                    self.output_b.as_device_ptr(),
                    output.as_device_ptr(),
                    self.output_size_a as u32,
                    self.output_size_b as u32
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
        _input: &DeviceBuffer<f32>, // Input is not needed as it is already buffered when the split happens
        output_gradient: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        let split_fn = self.module.get_function("split")?;

        let block_size = 256;
        let grid_size =
            ((self.output_size_a + self.output_size_b) as u32 + block_size - 1) / block_size;

        unsafe {
            cust::launch!(
                split_fn<<<grid_size, block_size, 0, real_stream>>>(
                    output_gradient.as_device_ptr(),
                    self.output_gradient_a.as_device_ptr(),
                    self.output_gradient_b.as_device_ptr(),
                    self.output_size_a as u32,
                    self.output_size_b as u32
                )
            )?;
        }

        self.layer_a.back_propagate(
            &mut self.input_gradient_a,
            &self.input_a,
            &self.output_gradient_a,
            stream,
        )?;
        self.layer_b.back_propagate(
            &mut self.input_gradient_b,
            &self.input_b,
            &self.output_gradient_b,
            stream,
        )?;

        let concat_fn = self.module.get_function("concat")?;

        let in_grid_size =
            ((self.input_size_a + self.input_size_b) as u32 + block_size - 1) / block_size;

        unsafe {
            cust::launch!(
                concat_fn<<<in_grid_size, block_size, 0, real_stream>>>(
                    self.input_gradient_a.as_device_ptr(),
                    self.input_gradient_b.as_device_ptr(),
                    input_gradient.as_device_ptr(),
                    self.input_size_a as u32,
                    self.input_size_b as u32
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
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.layer_a.optimize(stream)?;
        self.layer_b.optimize(stream)?;
        Ok(())
    }

    fn get_input_size(&self) -> usize {
        self.input_size_a + self.input_size_b
    }
    fn get_output_size(&self) -> usize {
        self.output_size_a + self.output_size_b
    }

    fn serialize_parameters(
        &self,
        writer: &mut dyn std::io::Write,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.layer_a.serialize_parameters(writer)?;
        self.layer_b.serialize_parameters(writer)?;
        Ok(())
    }

    fn deserialize_parameters(
        &mut self,
        reader: &mut dyn std::io::Read,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.layer_a.deserialize_parameters(reader)?;
        self.layer_b.deserialize_parameters(reader)?;
        Ok(())
    }
}
