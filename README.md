# faster_enhancer_rs

A pure Rust streaming speech enhancement library for FastEnhancer-Medium,
ported from [faster-enhancer.c](https://github.com/kdrkdrkdr/faster-enhancer.c/tree/6838816e9ea5369f94f64c57d900796aa9e6a2d5).
Uses `std` with no third-party dependencies.

- Requires x86-64 with AVX2, FMA, and F16C; selects available VNNI kernels at initialization.
- Processes 48 kHz mono audio in frames of 320 `f32` samples.
- Algorithmic delay: 704 samples (about 14.67 ms).
- No allocations, locks, or file I/O during frame processing.

```rust,no_run
use faster_enhancer_rs::Enhancer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut enhancer = Enhancer::new()?;
    enhancer.warm_up();

    let mut frame = [0.0_f32; 320]; // Fill with normalized PCM samples.
    enhancer.process_frame(&mut frame); // Enhanced audio replaces the input.
    enhancer.reset(); // Clear state before starting a new stream.
    Ok(())
}
```

Audio I/O and resampling are handled by the caller.
Multiple enhancers can share an `Arc<Model>` created with `Model::new()` via
`Enhancer::from_model`. Use `Model::from_weights` or `Enhancer::from_weights`
to load external weights.

Licensed under the [MIT License](LICENSE). See [NOTICE](NOTICE) for upstream code and model attribution.
