use cust::{device::Device, prelude::Context};

pub mod layers;

pub fn init_cuda() -> Result<Context, Box<dyn std::error::Error>> {
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let ctx = Context::new(device)?;
    Ok(ctx)
}
