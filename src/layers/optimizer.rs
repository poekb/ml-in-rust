use cust::{memory::DeviceBuffer, stream::Stream};

pub trait Optimizer {
    fn optimize(
        &self,
        parameters: &DeviceBuffer<f32>,
        gradient: &DeviceBuffer<f32>,
        gradient_count: usize,
        stream: Option<&Stream>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct StochasticGradientDescent {
    learning_rate: f32,
    module: cust::module::Module,
}

impl StochasticGradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        let module =
            cust::module::Module::from_ptx(include_str!("../kernels/optimizer.ptx"), &[])
                .expect("Failed to load optimizer PTX module");
        Self {
            learning_rate,
            module,
        }
    }
    pub fn boxed(learning_rate: f32) -> Box<Self> {
        Box::new(Self::new(learning_rate))
    }
}

impl Optimizer for StochasticGradientDescent {
    fn optimize(
        &self,
        parameters: &DeviceBuffer<f32>,
        gradient: &DeviceBuffer<f32>,
        gradient_count: usize,
        stream: Option<&Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if gradient_count == 0 {
            return Ok(());
        }
        let block_size = 256;
        let count = parameters.len();

        let grid_size = ((count as u32) + block_size - 1) / block_size;
        let optimize_fn = self.module.get_function("optimize")?;

        let real_stream = match stream {
            Some(s) => s,
            None => &Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        unsafe {
            cust::launch!(
                optimize_fn<<<grid_size, block_size, 0, real_stream>>>(
                    parameters.as_device_ptr(),
                    gradient.as_device_ptr(),
                    gradient_count as u32,
                    count as u32,
                    self.learning_rate,
                )
            )?;
        }
        Ok(())
    }
}
