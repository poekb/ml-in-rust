use cust::memory::DeviceBuffer;

use crate::layers::Layer;

static PTX: &str = include_str!("../kernels/layers/parallel.ptx");

pub enum MergeMode {
    Concatenate, // Stack outputs [a, b]
    Add,         // Element-wise addition a + b
    Multiply,    // Element-wise multiplication a * b
    Max,         // Element-wise maximum max(a, b)
}

pub struct ParallelLayer {
    input_size: usize,
    input_a: DeviceBuffer<f32>,
    input_b: DeviceBuffer<f32>,
    input_gradient_a: DeviceBuffer<f32>,
    input_gradient_b: DeviceBuffer<f32>,
    output_size_a: usize,
    output_size_b: usize,
    output_size: usize, // Total output size after merging
    output_a: DeviceBuffer<f32>,
    output_b: DeviceBuffer<f32>,
    output_gradient_a: DeviceBuffer<f32>,
    output_gradient_b: DeviceBuffer<f32>,
    layer_a: Box<dyn Layer>,
    layer_b: Box<dyn Layer>,
    merge_mode: MergeMode,
    module: cust::module::Module,
}

impl ParallelLayer {
    pub fn new(
        layer_a: Box<dyn Layer>,
        layer_b: Box<dyn Layer>,
        input_size: usize,
        merge_mode: MergeMode,
    ) -> Self {
        let output_size_a = layer_a.get_output_size();
        let output_size_b = layer_b.get_output_size();

        let output_size = match merge_mode {
            MergeMode::Concatenate => output_size_a + output_size_b,
            MergeMode::Add | MergeMode::Multiply | MergeMode::Max => {
                assert_eq!(
                    output_size_a, output_size_b,
                    "For Add/Multiply/Max merge modes, both layers must have the same output size"
                );
                output_size_a
            }
        };

        let input_a = DeviceBuffer::from_slice(&vec![0.0f32; input_size]).unwrap();
        let input_b = DeviceBuffer::from_slice(&vec![0.0f32; input_size]).unwrap();
        let input_gradient_a = DeviceBuffer::from_slice(&vec![0.0f32; input_size]).unwrap();
        let input_gradient_b = DeviceBuffer::from_slice(&vec![0.0f32; input_size]).unwrap();
        let output_a = DeviceBuffer::from_slice(&vec![0.0f32; output_size_a]).unwrap();
        let output_b = DeviceBuffer::from_slice(&vec![0.0f32; output_size_b]).unwrap();
        let output_gradient_a = DeviceBuffer::from_slice(&vec![0.0f32; output_size_a]).unwrap();
        let output_gradient_b = DeviceBuffer::from_slice(&vec![0.0f32; output_size_b]).unwrap();
        let module = cust::module::Module::from_ptx(PTX, &[]).expect("Failed to load PTX module");

        Self {
            input_size,
            input_a,
            input_b,
            input_gradient_a,
            input_gradient_b,
            output_size_a,
            output_size_b,
            output_size,
            output_a,
            output_b,
            output_gradient_a,
            output_gradient_b,
            layer_a,
            layer_b,
            merge_mode,
            module,
        }
    }

    pub fn boxed(
        layer_a: Box<dyn Layer>,
        layer_b: Box<dyn Layer>,
        input_size: usize,
        merge_mode: MergeMode,
    ) -> Box<Self> {
        Box::new(Self::new(layer_a, layer_b, input_size, merge_mode))
    }
}

impl Layer for ParallelLayer {
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

        let block_size = 256;
        let grid_size = (self.input_size as u32 + block_size - 1) / block_size;

        let duplicate_fn = self.module.get_function("duplicate_input")?;

        unsafe {
            cust::launch!(
                duplicate_fn<<<grid_size, block_size, 0, real_stream>>>(
                    input.as_device_ptr(),
                    self.input_a.as_device_ptr(),
                    self.input_b.as_device_ptr(),
                    self.input_size as u32
                )
            )?;
        }

        self.layer_a
            .forward(&self.input_a, &self.output_a, stream)?;
        self.layer_b
            .forward(&self.input_b, &self.output_b, stream)?;

        match self.merge_mode {
            MergeMode::Concatenate => {
                let concat_fn = self.module.get_function("concat")?;
                let grid_size = ((self.output_size_a + self.output_size_b) as u32 + block_size - 1)
                    / block_size;

                unsafe {
                    cust::launch!(
                        concat_fn<<<grid_size, block_size, 0, real_stream>>>(
                            self.output_a.as_device_ptr(),
                            self.output_b.as_device_ptr(),
                            output.as_device_ptr(),
                            self.output_size_a as u32,
                            self.output_size_b as u32
                        )
                    )?;
                }
            }
            MergeMode::Add => {
                let add_fn = self.module.get_function("element_wise_add")?;
                let grid_size = (self.output_size_a as u32 + block_size - 1) / block_size;

                unsafe {
                    cust::launch!(
                        add_fn<<<grid_size, block_size, 0, real_stream>>>(
                            self.output_a.as_device_ptr(),
                            self.output_b.as_device_ptr(),
                            output.as_device_ptr(),
                            self.output_size_a as u32
                        )
                    )?;
                }
            }
            MergeMode::Multiply => {
                let mul_fn = self.module.get_function("element_wise_multiply")?;
                let grid_size = (self.output_size_a as u32 + block_size - 1) / block_size;

                unsafe {
                    cust::launch!(
                        mul_fn<<<grid_size, block_size, 0, real_stream>>>(
                            self.output_a.as_device_ptr(),
                            self.output_b.as_device_ptr(),
                            output.as_device_ptr(),
                            self.output_size_a as u32
                        )
                    )?;
                }
            }
            MergeMode::Max => {
                let max_fn = self.module.get_function("element_wise_max")?;
                let grid_size = (self.output_size_a as u32 + block_size - 1) / block_size;

                unsafe {
                    cust::launch!(
                        max_fn<<<grid_size, block_size, 0, real_stream>>>(
                            self.output_a.as_device_ptr(),
                            self.output_b.as_device_ptr(),
                            output.as_device_ptr(),
                            self.output_size_a as u32
                        )
                    )?;
                }
            }
        }

        if stream.is_none() {
            real_stream.synchronize()?;
        }

        Ok(())
    }

    fn back_propagate(
        &mut self,
        input_gradient: &DeviceBuffer<f32>,
        _input: &DeviceBuffer<f32>, // Input is buffered when di
        output_gradient: &DeviceBuffer<f32>,
        stream: Option<&cust::stream::Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let real_stream = match stream {
            Some(s) => s,
            None => &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        let block_size = 256;

        match self.merge_mode {
            MergeMode::Concatenate => {
                let split_fn = self.module.get_function("split")?;
                let grid_size = ((self.output_size_a + self.output_size_b) as u32 + block_size - 1)
                    / block_size;

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
            }
            MergeMode::Add => {
                let grid_size = (self.output_size_a as u32 + block_size - 1) / block_size;
                let duplicate_fn = self.module.get_function("duplicate_input")?;

                unsafe {
                    cust::launch!(
                        duplicate_fn<<<grid_size, block_size, 0, real_stream>>>(
                            output_gradient.as_device_ptr(),
                            self.output_gradient_a.as_device_ptr(),
                            self.output_gradient_b.as_device_ptr(),
                            self.output_size_a as u32
                        )
                    )?;
                }
            }
            MergeMode::Multiply => {
                let mul_gradient_fn = self.module.get_function("multiply_gradient_distribution")?;
                let grid_size = (self.output_size_a as u32 + block_size - 1) / block_size;

                unsafe {
                    cust::launch!(
                        mul_gradient_fn<<<grid_size, block_size, 0, real_stream>>>(
                            output_gradient.as_device_ptr(),
                            self.output_a.as_device_ptr(),
                            self.output_b.as_device_ptr(),
                            self.output_gradient_a.as_device_ptr(),
                            self.output_gradient_b.as_device_ptr(),
                            self.output_size_a as u32
                        )
                    )?;
                }
            }
            MergeMode::Max => {
                let max_gradient_fn = self.module.get_function("max_gradient_distribution")?;
                let grid_size = (self.output_size_a as u32 + block_size - 1) / block_size;

                unsafe {
                    cust::launch!(
                        max_gradient_fn<<<grid_size, block_size, 0, real_stream>>>(
                            output_gradient.as_device_ptr(),
                            self.output_a.as_device_ptr(),
                            self.output_b.as_device_ptr(),
                            self.output_gradient_a.as_device_ptr(),
                            self.output_gradient_b.as_device_ptr(),
                            self.output_size_a as u32
                        )
                    )?;
                }
            }
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

        let add_fn = self.module.get_function("element_wise_add")?;
        let in_grid_size = (self.input_size as u32 + block_size - 1) / block_size;

        unsafe {
            cust::launch!(
                add_fn<<<in_grid_size, block_size, 0, real_stream>>>(
                    self.input_gradient_a.as_device_ptr(),
                    self.input_gradient_b.as_device_ptr(),
                    input_gradient.as_device_ptr(),
                    self.input_size as u32
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
        self.input_size
    }

    fn get_output_size(&self) -> usize {
        self.output_size
    }
}
