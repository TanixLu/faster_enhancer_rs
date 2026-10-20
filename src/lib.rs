//! FastEnhancer-Medium streaming inference on x86-64. No third-party dependencies.
//! Each frame contains 320 mono f32 samples at 48 kHz.
//! Model weights are embedded; initialize with `Enhancer::new()`.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_arch = "x86_64"))]
compile_error!("faster_enhancer_rs supports x86-64 only (AVX2 + FMA + F16C)");

#[cfg(target_arch = "x86_64")]
mod engine;
#[cfg(target_arch = "x86_64")]
mod fft;
mod memory;
#[cfg(target_arch = "x86_64")]
mod model;
#[cfg(target_arch = "x86_64")]
mod simd;
#[cfg(target_arch = "x86_64")]
pub use engine::Enhancer;
#[cfg(target_arch = "x86_64")]
pub use model::Model;

pub const SAMPLE_RATE: u32 = 48_000;
pub const FRAME_SIZE: usize = 320;
pub const ALGORITHM_DELAY: usize = 704;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitError {
    UnsupportedCpu,
    InvalidLength,
    InvalidMagic,
    UnsupportedVersion,
    InvalidWeight,
    Allocation,
}
impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UnsupportedCpu => "x86-64 with AVX2, FMA and F16C is required",
            Self::InvalidLength => "incorrect FastEnhancer-Medium weight blob length",
            Self::InvalidMagic => "expected FM_W8_03 weight blob",
            Self::UnsupportedVersion => "unsupported weight version",
            Self::InvalidWeight => "invalid or out-of-range model weight",
            Self::Allocation => "memory allocation failed",
        })
    }
}
impl std::error::Error for InitError {}

/// Instruction set used by the integer matrix kernels. All tiers use the same
/// AVX2 FFT and floating-point epilogues to preserve their operation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTier {
    Avx2,
    AvxVnni,
    Avx512Vnni,
}

/// A live allocation, exposed for caller-managed page residency. The address
/// of numeric allocations remains valid while its owner is alive. Descriptor
/// addresses also require that the Model/Enhancer does not move after querying.
/// Place the instance at its final address first; the library never pins pages.
#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion {
    pub address: *const u8,
    pub len: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Statistics {
    pub frames: u64,
    pub sanitized_samples: u64,
}
