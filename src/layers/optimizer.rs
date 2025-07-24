use cust::{memory::DeviceBuffer, stream::Stream};

pub trait Optimizer {
    fn instance(&self, param_count: usize) -> Box<dyn OptimizerImpl>;
}

pub trait OptimizerImpl {
    fn optimize(
        &mut self,
        parameters: &DeviceBuffer<f32>,
        gradient: &DeviceBuffer<f32>,
        gradient_count: usize,
        stream: Option<&Stream>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct StochasticGradientDescent {
    learning_rate: f32,
}

impl StochasticGradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
    pub fn boxed(learning_rate: f32) -> Box<Self> {
        Box::new(Self::new(learning_rate))
    }
}

impl Optimizer for StochasticGradientDescent {
    fn instance(&self, _param_count: usize) -> Box<dyn OptimizerImpl> {
        let module = cust::module::Module::from_ptx(include_str!("../kernels/optimizer.ptx"), &[])
            .expect("Failed to load optimizer PTX module");
        Box::new(StochasticGradientDescentOpt {
            learning_rate: self.learning_rate,
            module,
        })
    }
}

struct StochasticGradientDescentOpt {
    learning_rate: f32,
    module: cust::module::Module,
}

impl OptimizerImpl for StochasticGradientDescentOpt {
    fn optimize(
        &mut self,
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

// Impl the Adam optimizer.
pub struct AdamOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }
    pub fn boxed(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Box<Self> {
        Box::new(Self::new(learning_rate, beta1, beta2, epsilon))
    }
}

impl Optimizer for AdamOptimizer {
    fn instance(&self, param_count: usize) -> Box<dyn OptimizerImpl> {
        let module = cust::module::Module::from_ptx(include_str!("../kernels/optimizer.ptx"), &[])
            .expect("Failed to load Adam optimizer PTX module");
        Box::new(AdamOptimizerImpl {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            module,
            m: DeviceBuffer::zeroed(param_count).unwrap(),
            v: DeviceBuffer::zeroed(param_count).unwrap(),
            t: 1,
        })
    }
}

struct AdamOptimizerImpl {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    module: cust::module::Module,
    m: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    t: u32,
}

impl OptimizerImpl for AdamOptimizerImpl {
    fn optimize(
        &mut self,
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
        let optimize_fn = self.module.get_function("adam_optimize")?;

        let real_stream = match stream {
            Some(s) => s,
            None => &Stream::new(cust::stream::StreamFlags::DEFAULT, None)?,
        };

        unsafe {
            cust::launch!(
                optimize_fn<<<grid_size, block_size, 0, real_stream>>>(
                    parameters.as_device_ptr(),
                    gradient.as_device_ptr(),
                    self.m.as_device_ptr(),
                    self.v.as_device_ptr(),
                    self.learning_rate,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    gradient_count as u32,
                    count as u32,
                    self.t
                )
            )?;
        }

        if stream.is_none() {
            real_stream.synchronize()?;
        }

        self.t += 1;
        Ok(())
    }
}

pub struct NoOpOptimizer;

impl NoOpOptimizer {
    pub fn new() -> Self {
        Self {}
    }
    pub fn boxed() -> Box<Self> {
        Box::new(Self::new())
    }
}

impl Optimizer for NoOpOptimizer {
    fn instance(&self, _param_count: usize) -> Box<dyn OptimizerImpl> {
        Box::new(NoOpOptimizerImpl)
    }
}
struct NoOpOptimizerImpl;

impl OptimizerImpl for NoOpOptimizerImpl {
    fn optimize(
        &mut self,
        _parameters: &DeviceBuffer<f32>,
        _gradient: &DeviceBuffer<f32>,
        _gradient_count: usize,
        _stream: Option<&Stream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
