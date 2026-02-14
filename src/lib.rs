pub mod autograd;
pub mod ffi;
pub mod nn;
pub mod ops;
pub mod tensor;

pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    // Integration tests could go here or in tests/ directory
}

