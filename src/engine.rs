//! Fixed inference graph. Unsafe is confined here and in the microkernels:
//! dimensions come exclusively from the validated Medium model. Arena ranges
//! below have fixed capacities; reused ranges have nonoverlapping lifetimes.
#![allow(unsafe_op_in_unsafe_fn)]
// Fixed graph dimensions and register tiles mirror the C microkernel ABI.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use crate::{
    InitError, MemoryRegion, Model, SimdTier, Statistics, fft,
    memory::Buffer,
    model::Matrix,
    simd::{self, Quant},
};
use core::arch::x86_64::*;
use std::{ptr, sync::Arc};
const FC: usize = 128 * 96;
const RF: usize = 72 * 72;

struct State {
    analysis: Buffer<f32>,
    synthesis: Buffer<f32>,
    hidden: Buffer<u16>,
    stats: Statistics,
}
struct Workspace {
    fft: Buffer<f32>,
    time: Buffer<f32>,
    bins: Buffer<f32>,
    spec: Buffer<f32>,
    masked: Buffer<f32>,
    a: Buffer<f32>,
    b: Buffer<f32>,
    skips: Buffer<u16>,
    // 122880 bytes: Winograd U+dq OR attention qkv/scores/output/packs.
    aux: Buffer<i32>,
    aq: Buffer<i8>,
    wide: Buffer<i16>,
    c32: Buffer<i32>,
    pair: Buffer<i8>,
    // GRU tile, dequant coefficients and float hidden output. Never stack arrays.
    gru_i: Buffer<i32>,
    gru_h: Buffer<i32>,
    gru_out: Buffer<f32>,
    coeff: Buffer<f32>,
    im2col: Buffer<f32>,
}
impl State {
    fn new() -> Result<Self, InitError> {
        Ok(Self {
            analysis: Buffer::new(704)?,
            synthesis: Buffer::new(704)?,
            hidden: Buffer::new(4 * RF)?,
            stats: Statistics::default(),
        })
    }
    fn reset(&mut self) {
        self.analysis.fill(0.);
        self.synthesis.fill(0.);
        self.hidden.fill(0);
        self.stats = Statistics::default();
    }
    fn regions(&self) -> [MemoryRegion; 3] {
        [
            self.analysis.region(),
            self.synthesis.region(),
            self.hidden.region(),
        ]
    }
}
impl Workspace {
    fn new() -> Result<Self, InitError> {
        Ok(Self {
            fft: Buffer::new(4096)?,
            time: Buffer::new(1024)?,
            bins: Buffer::new(1026)?,
            spec: Buffer::new(1024)?,
            masked: Buffer::new(1024)?,
            a: Buffer::new(FC)?,
            b: Buffer::new(FC)?,
            skips: Buffer::new(4 * FC)?,
            aux: Buffer::new(30720)?,
            aq: Buffer::new(2 * FC)?,
            wide: Buffer::new(2 * FC)?,
            c32: Buffer::new(72 * 72)?,
            pair: Buffer::new(72 * 24)?,
            gru_i: Buffer::new(8 * 216)?,
            gru_h: Buffer::new(8 * 216)?,
            gru_out: Buffer::new(RF)?,
            coeff: Buffer::new(4 * 216)?,
            im2col: Buffer::new(128 * 16)?,
        })
    }
    fn regions(&self) -> [MemoryRegion; 18] {
        [
            self.fft.region(),
            self.time.region(),
            self.bins.region(),
            self.spec.region(),
            self.masked.region(),
            self.a.region(),
            self.b.region(),
            self.skips.region(),
            self.aux.region(),
            self.aq.region(),
            self.wide.region(),
            self.c32.region(),
            self.pair.region(),
            self.gru_i.region(),
            self.gru_h.region(),
            self.gru_out.region(),
            self.coeff.region(),
            self.im2col.region(),
        ]
    }
}

/// An independently owned stream. `&mut self` prevents concurrent processing
/// of the same stream; different instances share only immutable Model data.
pub struct Enhancer {
    model: Arc<Model>,
    state: State,
    workspace: Workspace,
    tier: SimdTier,
    dot: simd::Dot,
}
impl Enhancer {
    /// Create a stream using the model weights embedded in the library.
    pub fn new() -> Result<Self, InitError> {
        Self::from_model(Arc::new(Model::new()?))
    }
    /// Create a stream using external model weights.
    pub fn from_weights(weights: &[u8]) -> Result<Self, InitError> {
        Self::from_model(Arc::new(Model::from_weights(weights)?))
    }
    pub fn from_model(model: Arc<Model>) -> Result<Self, InitError> {
        Self::with_tier(model, None)
    }
    /// Force a supported tier for reproducible comparisons; None selects the
    /// highest supported tier. Feature detection never occurs in process_frame.
    pub fn with_tier(model: Arc<Model>, tier: Option<SimdTier>) -> Result<Self, InitError> {
        let tier = simd::select(tier)?;
        let dot: simd::Dot = match tier {
            SimdTier::Avx2 => simd::dot_avx2,
            SimdTier::AvxVnni => simd::dot_vnni,
            SimdTier::Avx512Vnni => simd::dot_vnni512,
        };
        Ok(Self {
            model,
            state: State::new()?,
            workspace: Workspace::new()?,
            tier,
            dot,
        })
    }
    pub const fn sample_rate(&self) -> u32 {
        crate::SAMPLE_RATE
    }
    pub const fn frame_size(&self) -> usize {
        crate::FRAME_SIZE
    }
    pub const fn algorithm_delay(&self) -> usize {
        crate::ALGORITHM_DELAY
    }
    pub fn simd_tier(&self) -> SimdTier {
        self.tier
    }
    pub fn statistics(&self) -> Statistics {
        self.state.stats
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    pub fn workspace_size(&self) -> usize {
        self.workspace.regions().iter().map(|r| r.len).sum()
    }
    pub fn state_size(&self) -> usize {
        self.state.regions().iter().map(|r| r.len).sum::<usize>() + size_of::<Statistics>()
    }
    /// Query outside the audio callback. Includes all large model/state/scratch
    /// allocations and descriptors. Place the Enhancer at its final address
    /// before querying; moving it invalidates its descriptor region. Model
    /// regions appear in every sharing instance's result.
    pub fn memory_regions(&self) -> Vec<MemoryRegion> {
        let mut r = self.model.memory_regions();
        r.push(MemoryRegion {
            address: (self as *const Self).cast(),
            len: size_of::<Self>(),
        });
        r.extend(self.state.regions());
        r.extend(self.workspace.regions());
        r
    }
    pub fn reset(&mut self) {
        self.state.reset();
    }
    /// Touch the full inference path, then reset persistent state and counters.
    /// No process-wide floating-point control register is modified.
    pub fn warm_up(&mut self) {
        for _ in 0..128 {
            let mut frame = [0.; 320];
            self.process_frame(&mut frame);
        }
        self.reset();
    }
    /// Process exactly one frame in place, without allocation, I/O, locks,
    /// dispatch initialization or ownership changes. NaNs/infinities, subnormal
    /// samples are sanitized at the input boundary.
    /// Use normalized PCM to preserve the reference's finite-input contract.
    pub fn process_frame(&mut self, frame: &mut [f32; 320]) {
        for x in frame.iter_mut() {
            let bits = x.to_bits();
            let exponent = bits & 0x7f800000;
            let v = if exponent == 0 {
                f32::from_bits(bits & 0x80000000)
            } else if exponent == 0x7f800000 {
                0.
            } else {
                *x
            };
            if v.to_bits() != bits {
                self.state.stats.sanitized_samples =
                    self.state.stats.sanitized_samples.saturating_add(1);
            }
            *x = v;
        }
        // SAFETY: initialization checked AVX2+FMA+F16C and the selected integer
        // tier. All buffers are uniquely owned, initialized, and fixed-sized.
        unsafe {
            run(
                &self.model,
                &mut self.state,
                &mut self.workspace,
                self.tier,
                self.dot,
                frame.as_mut_ptr(),
            );
        }
        self.state.stats.frames = self.state.stats.frames.saturating_add(1);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matmul(
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    m: usize,
    matrix: &Matrix,
    aq: *const i8,
    c: *mut i32,
    post: *const simd::Post,
) {
    let b = if tier == SimdTier::Avx2 {
        simd::expand(aq, w.wide.as_ptr() as *mut i16, m * matrix.k);
        matrix.pair.as_ptr()
    } else {
        matrix.q.as_ptr()
    };
    dot(m, matrix.n, matrix.k, aq, b, w.wide.as_ptr(), c, post);
}

/// The caller holds unique Workspace access throughout each call. Raw arena
/// pointers are used to express the graph's disjoint and reused ranges.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn linear(
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    m: usize,
    matrix: &Matrix,
    input: *const f32,
    half: *const u16,
    transpose: bool,
    out: *mut f32,
    silu: bool,
    accumulate: bool,
) {
    let (min, max) = simd::minmax(input, half, m * matrix.k);
    let q = simd::qparams(min, max);
    simd::quantize(
        input,
        half,
        m,
        matrix.k,
        transpose,
        w.aq.as_ptr() as *mut i8,
        q,
    );
    linear_quantized(
        w,
        tier,
        dot,
        m,
        matrix,
        w.aq.as_ptr(),
        q,
        out,
        silu,
        accumulate,
    );
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn coefficients(
    matrix: &Matrix,
    q: Quant,
    scale: *mut f32,
    bias: *mut f32,
    include_bias: bool,
) {
    for n in (0..matrix.n).step_by(8) {
        let cs = _mm256_mul_ps(
            _mm256_set1_ps(q.scale),
            _mm256_loadu_ps(matrix.scales.as_ptr().add(n)),
        );
        _mm256_storeu_ps(scale.add(n), cs);
        let b = if include_bias && matrix.has_bias {
            _mm256_loadu_ps(matrix.bias.as_ptr().add(n))
        } else {
            _mm256_setzero_ps()
        };
        // For standard linear zp=128 bypasses the correction, exactly as C.
        let eff = if q.zp == 128 {
            b
        } else {
            _mm256_fmadd_ps(
                _mm256_mul_ps(cs, _mm256_set1_ps((128 - q.zp) as f32)),
                _mm256_cvtepi32_ps(_mm256_loadu_si256(matrix.sums.as_ptr().add(n).cast())),
                b,
            )
        };
        _mm256_storeu_ps(bias.add(n), eff);
    }
}
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn linear_quantized(
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    m: usize,
    matrix: &Matrix,
    aq: *const i8,
    q: Quant,
    out: *mut f32,
    silu: bool,
    accumulate: bool,
) {
    let cs = w.coeff.as_ptr() as *mut f32;
    let bs = cs.add(216);
    coefficients(matrix, q, cs, bs, true);
    let post = simd::Post {
        scale: cs,
        bias: if matrix.has_bias || q.zp != 128 {
            bs
        } else {
            ptr::null()
        },
        out,
        silu,
        accumulate,
    };
    matmul(w, tier, dot, m, matrix, aq, ptr::null_mut(), &post);
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn concat(
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    matrix: &Matrix,
    input: *const f32,
    skip: *const u16,
    out: *mut f32,
) {
    let a = simd::minmax(input, ptr::null(), FC);
    let b = simd::minmax(ptr::null(), skip, FC);
    let q = simd::qparams(a.0.min(b.0), a.1.max(b.1));
    let aq = w.aq.as_ptr() as *mut i8;
    for r in 0..128 {
        simd::quantize(
            input.add(r * 96),
            ptr::null(),
            1,
            96,
            false,
            aq.add(r * 192),
            q,
        );
        simd::quantize(
            ptr::null(),
            skip.add(r * 96),
            1,
            96,
            false,
            aq.add(r * 192 + 96),
            q,
        );
    }
    linear_quantized(w, tier, dot, 128, matrix, aq, q, out, true, false);
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn encoder_pre(model: &Model, w: &Workspace, out: *mut f32) {
    let input = w.spec.as_ptr();
    let col = w.im2col.as_ptr() as *mut f32;
    for f in 0..128 {
        for i in 0..8 {
            for tap in 0..2 {
                let p = ((f + tap) * 4 + i / 2) as isize - 2;
                *col.add(f * 16 + i * 2 + tap) = if (0..512).contains(&p) {
                    *input.add(i % 2 * 512 + p as usize)
                } else {
                    0.
                };
            }
        }
    }
    for r in (0..128).step_by(8) {
        for n in (0..96).step_by(8) {
            let mut acc = [_mm256_loadu_ps(model.enc_bias.as_ptr().add(n)); 8];
            for k in 0..16 {
                let b = _mm256_loadu_ps(model.enc_weight.as_ptr().add(n * 16 + k * 8));
                for j in 0..8 {
                    acc[j] = _mm256_fmadd_ps(_mm256_set1_ps(*col.add((r + j) * 16 + k)), b, acc[j]);
                }
            }
            for j in 0..8 {
                _mm256_storeu_ps(out.add((r + j) * 96 + n), acc[j]);
            }
        }
    }
    let skip = w.skips.as_ptr() as *mut u16;
    for i in (0..FC).step_by(8) {
        let v = _mm256_loadu_ps(out.add(i));
        let y = simd::normal(_mm256_mul_ps(v, simd::sigmoid(v)));
        _mm256_storeu_ps(out.add(i), y);
        _mm_storeu_si128(skip.add(i).cast(), _mm256_cvtps_ph::<0>(y));
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wino(
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    layer: &crate::model::Layer,
    input: *const f32,
    out: *mut f32,
    skip: *mut u16,
) {
    // Model construction guarantees all six k3 layers have four derived matrices.
    let Some(matrices) = &layer.wino else {
        return;
    };
    let u = w.aux.as_ptr() as *mut i32;
    let dq = u.add(4 * 64 * 96).cast::<i8>();
    let mut maxima = [_mm256_setzero_ps(); 4];
    for t in 0..64 {
        for c in (0..96).step_by(8) {
            let p1 = _mm256_loadu_ps(input.add((t * 2) * 96 + c));
            let p2 = _mm256_loadu_ps(input.add((t * 2 + 1) * 96 + c));
            let p0 = if t > 0 {
                _mm256_loadu_ps(input.add((t * 2 - 1) * 96 + c))
            } else {
                _mm256_setzero_ps()
            };
            let p3 = if t < 63 {
                _mm256_loadu_ps(input.add((t * 2 + 2) * 96 + c))
            } else {
                _mm256_setzero_ps()
            };
            let d = [
                _mm256_sub_ps(p0, p2),
                _mm256_add_ps(p1, p2),
                _mm256_sub_ps(p2, p1),
                _mm256_sub_ps(p1, p3),
            ];
            for j in 0..4 {
                maxima[j] = _mm256_max_ps(
                    maxima[j],
                    _mm256_and_ps(d[j], _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))),
                );
            }
        }
    }
    let mut scale = [0.; 4];
    let mut inv = [0.; 4];
    for j in 0..4 {
        let mut lanes = [0.; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), maxima[j]);
        let mut max = lanes.into_iter().fold(0., f32::max);
        if max == 0. {
            max = 1.;
        }
        scale[j] = max / 127.;
        inv[j] = 127. / max;
    }
    for t in 0..64 {
        for c in (0..96).step_by(8) {
            let p1 = _mm256_loadu_ps(input.add(t * 2 * 96 + c));
            let p2 = _mm256_loadu_ps(input.add((t * 2 + 1) * 96 + c));
            let p0 = if t > 0 {
                _mm256_loadu_ps(input.add((t * 2 - 1) * 96 + c))
            } else {
                _mm256_setzero_ps()
            };
            let p3 = if t < 63 {
                _mm256_loadu_ps(input.add((t * 2 + 2) * 96 + c))
            } else {
                _mm256_setzero_ps()
            };
            let d = [
                _mm256_sub_ps(p0, p2),
                _mm256_add_ps(p1, p2),
                _mm256_sub_ps(p2, p1),
                _mm256_sub_ps(p1, p3),
            ];
            for j in 0..4 {
                simd::pack8(
                    _mm256_cvtps_epi32(_mm256_mul_ps(d[j], _mm256_set1_ps(inv[j]))),
                    dq.add(j * 64 * 96 + t * 96 + c),
                );
            }
        }
    }
    let coeff = w.coeff.as_ptr() as *mut f32;
    for j in 0..4 {
        matmul(
            w,
            tier,
            dot,
            64,
            &matrices[j],
            dq.add(j * 64 * 96),
            u.add(j * 64 * 96),
            ptr::null(),
        );
        for n in (0..96).step_by(8) {
            _mm256_storeu_ps(
                coeff.add(j * 96 + n),
                _mm256_mul_ps(
                    _mm256_set1_ps(scale[j]),
                    _mm256_loadu_ps(matrices[j].scales.as_ptr().add(n)),
                ),
            );
        }
    }
    for t in 0..64 {
        for n in (0..96).step_by(8) {
            let mut v = [_mm256_setzero_ps(); 4];
            for j in 0..4 {
                v[j] = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(u.add(j * 64 * 96 + t * 96 + n).cast())),
                    _mm256_loadu_ps(coeff.add(j * 96 + n)),
                );
            }
            let b = _mm256_loadu_ps(layer.matrix.bias.as_ptr().add(n));
            let ye = _mm256_add_ps(_mm256_add_ps(v[0], v[1]), _mm256_add_ps(v[2], b));
            let yo = _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(v[1], v[2]), v[3]), b);
            let ye = simd::normal(_mm256_mul_ps(ye, simd::sigmoid(ye)));
            let yo = simd::normal(_mm256_mul_ps(yo, simd::sigmoid(yo)));
            let e = t * 2 * 96 + n;
            let o = (t * 2 + 1) * 96 + n;
            _mm256_storeu_ps(out.add(e), ye);
            _mm256_storeu_ps(out.add(o), yo);
            if !skip.is_null() {
                _mm_storeu_si128(skip.add(e).cast(), _mm256_cvtps_ph::<0>(ye));
                _mm_storeu_si128(skip.add(o).cast(), _mm256_cvtps_ph::<0>(yo));
            }
        }
    }
}

/// GRU computes both integer projections for an eight-frequency tile, then
/// immediately dequantizes and evaluates all gates. Tile scratch is reused;
/// no full-frame gate tensors, hidden unpack, or fp16 packing passes exist.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gru(
    model: &Model,
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    block: usize,
    input: *const f32,
    hidden: *mut u16,
) {
    let ih = &model.layers[5 + block * 5].matrix;
    let hh = &model.layers[6 + block * 5].matrix;
    let a = simd::minmax(input, ptr::null(), RF);
    let h = simd::minmax(ptr::null(), hidden, RF);
    let qx = simd::qparams(a.0, a.1);
    let qh = simd::qparams(h.0, h.1);
    let aq = w.aq.as_ptr() as *mut i8;
    simd::quantize(input, ptr::null(), 72, 72, false, aq, qx);
    simd::quantize(ptr::null(), hidden, 72, 72, false, aq.add(RF), qh);
    let cs_i = w.coeff.as_ptr() as *mut f32;
    let bs_i = cs_i.add(216);
    let cs_h = cs_i.add(432);
    let bs_h = cs_i.add(648);
    coefficients(ih, qx, cs_i, bs_i, false);
    coefficients(hh, qh, cs_h, bs_h, false);
    for row in (0..72).step_by(8) {
        matmul(
            w,
            tier,
            dot,
            8,
            ih,
            aq.add(row * 72),
            w.gru_i.as_ptr() as *mut i32,
            ptr::null(),
        );
        matmul(
            w,
            tier,
            dot,
            8,
            hh,
            aq.add(RF + row * 72),
            w.gru_h.as_ptr() as *mut i32,
            ptr::null(),
        );
        for r in 0..8 {
            for n in (0..72).step_by(8) {
                let mut vi = [_mm256_setzero_ps(); 3];
                let mut vh = [_mm256_setzero_ps(); 3];
                for g in 0..3 {
                    let c = g * 72 + n;
                    let ix = r * 216 + c;
                    vi[g] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(w.gru_i.as_ptr().add(ix).cast())),
                        _mm256_loadu_ps(cs_i.add(c)),
                        _mm256_loadu_ps(bs_i.add(c)),
                    );
                    vh[g] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(w.gru_h.as_ptr().add(ix).cast())),
                        _mm256_loadu_ps(cs_h.add(c)),
                        _mm256_loadu_ps(bs_h.add(c)),
                    );
                }
                let br = _mm256_loadu_ps(model.combined_bias.as_ptr().add(block * 216 + n));
                let bz = _mm256_loadu_ps(model.combined_bias.as_ptr().add(block * 216 + 72 + n));
                let reset = simd::sigmoid(_mm256_add_ps(_mm256_add_ps(vi[0], vh[0]), br));
                let update = simd::sigmoid(_mm256_add_ps(_mm256_add_ps(vi[1], vh[1]), bz));
                let ni = _mm256_add_ps(vi[2], _mm256_loadu_ps(ih.bias.as_ptr().add(144 + n)));
                let nh = _mm256_add_ps(vh[2], _mm256_loadu_ps(hh.bias.as_ptr().add(144 + n)));
                let candidate = simd::tanh(_mm256_fmadd_ps(reset, nh, ni));
                let ix = (row + r) * 72 + n;
                let old = _mm256_cvtph_ps(_mm_loadu_si128(hidden.add(ix).cast()));
                let new = simd::normal(_mm256_fmadd_ps(
                    update,
                    _mm256_sub_ps(old, candidate),
                    candidate,
                ));
                _mm256_storeu_ps((w.gru_out.as_ptr() as *mut f32).add(ix), new);
                _mm_storeu_si128(hidden.add(ix).cast(), _mm256_cvtps_ph::<0>(new));
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn softmax(c32: *const i32, scale: f32, scratch: *mut f32, out: *mut i8) {
    for r in 0..72 {
        let mut sum = _mm256_setzero_ps();
        for n in (0..72).step_by(8) {
            let x = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(c32.add(r * 72 + n).cast())),
                _mm256_set1_ps(scale),
            );
            let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.)), _mm256_set1_ps(88.));
            let bits = _mm256_cvtps_epi32(_mm256_fmadd_ps(
                x,
                _mm256_set1_ps(12102203.),
                _mm256_set1_ps(1064866805.),
            ));
            let e = _mm256_castsi256_ps(bits);
            _mm256_storeu_ps(scratch.add(n), e);
            sum = _mm256_add_ps(sum, e);
        }
        // Match fe_hsum256's exact reduction order.
        let v = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
        let sh = _mm_movehdup_ps(v);
        let r1 = _mm_add_ps(v, sh);
        let r2 = _mm_movehl_ps(sh, r1);
        let total = _mm_cvtss_f32(_mm_add_ss(r1, r2));
        let factor = _mm256_set1_ps((1. / total) * 127.);
        for n in (0..72).step_by(8) {
            simd::pack8(
                _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(scratch.add(n)), factor)),
                out.add(r * 72 + n),
            );
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn attention(
    model: &Model,
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    block: usize,
    input: *mut f32,
) {
    let qkv = w.aux.as_ptr() as *mut f32;
    let score = qkv.add(15552);
    let attn = score.add(72);
    let qkv_q = attn.add(RF).cast::<i8>();
    let q_all = qkv_q.add(15552);
    let k_all = q_all.add(4 * 72 * 20);
    let v_all = k_all.add(4 * 72 * 20);
    let sq = v_all.add(4 * 72 * 24);
    linear(
        w,
        tier,
        dot,
        72,
        &model.layers[8 + block * 5].matrix,
        input,
        ptr::null(),
        false,
        qkv,
        false,
        false,
    );
    let range = simd::minmax(qkv, ptr::null(), 15552);
    let mut max = range.0.abs().max(range.1.abs());
    if max == 0. {
        max = 1.;
    }
    let qs = max / 127.;
    simd::quantize(
        qkv,
        ptr::null(),
        72,
        216,
        false,
        qkv_q,
        Quant { scale: qs, zp: 128 },
    );
    // Fused Q/K/V extraction and packing from each quantized QKV row.
    ptr::write_bytes(q_all, 0, 4 * 72 * 20);
    ptr::write_bytes(k_all, 0, 4 * 72 * 20);
    ptr::write_bytes(v_all, 0, 4 * 72 * 24);
    for r in 0..72 {
        for h in 0..4 {
            for d in 0..18 {
                let src = qkv_q.add(r * 216 + h * 54);
                *q_all.add(h * 72 * 20 + r * 20 + d) = *src.add(d);
                *k_all.add(h * 72 * 20 + crate::model::index(r, d, 20)) = *src.add(18 + d);
                *v_all.add(h * 72 * 24 + crate::model::index(d, r, 72)) = *src.add(36 + d);
            }
        }
    }
    let scale = qs * qs * (1. / 18.0_f32.sqrt());
    for h in 0..4 {
        dynamic_dot(
            w,
            tier,
            dot,
            72,
            72,
            20,
            q_all.add(h * 72 * 20),
            k_all.add(h * 72 * 20),
            w.c32.as_ptr() as *mut i32,
        );
        softmax(w.c32.as_ptr(), scale, score, sq);
        dynamic_dot(
            w,
            tier,
            dot,
            72,
            24,
            72,
            sq,
            v_all.add(h * 72 * 24),
            w.c32.as_ptr() as *mut i32,
        );
        let cs = (1. / 127.) * qs;
        for r in 0..72 {
            for d in 0..18 {
                *attn.add(r * 72 + h * 18 + d) = *w.c32.as_ptr().add(r * 24 + d) as f32 * cs;
            }
        }
    }
    linear(
        w,
        tier,
        dot,
        72,
        &model.layers[9 + block * 5].matrix,
        attn,
        ptr::null(),
        false,
        input,
        false,
        true,
    );
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dynamic_dot(
    w: &Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    b: *const i8,
    c: *mut i32,
) {
    let packed = if tier == SimdTier::Avx2 {
        let dst = w.pair.as_ptr() as *mut i8;
        for bn in 0..n / 8 {
            for g in 0..k / 4 {
                for lane in 0..8 {
                    for j in 0..4 {
                        *dst.add(bn * k * 8 + (g * 2 + j / 2) * 16 + lane * 2 + j % 2) =
                            *b.add(bn * k * 8 + g * 32 + lane * 4 + j);
                    }
                }
            }
        }
        simd::expand(a, w.wide.as_ptr() as *mut i16, m * k);
        dst as *const i8
    } else {
        b
    };
    dot(m, n, k, a, packed, w.wide.as_ptr(), c, ptr::null());
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn upsample(model: &Model, input: *const f32, out: *mut f32) {
    for o in 0..2 {
        for f in 0..512 {
            *out.add(o * 512 + f) = model.up_bias[o];
        }
    }
    for i in 0..96 {
        let ww0 = model.up_weight.as_ptr().add(i * 16);
        let ww1 = ww0.add(8);
        for f in 0..128 {
            let x = _mm_set1_ps(*input.add(f * 96 + i));
            let base = (f * 4) as isize - 2;
            if f == 0 || f == 127 {
                let x = *input.add(f * 96 + i);
                for k in 0..8 {
                    let p = base + k as isize;
                    if (0..512).contains(&p) {
                        *out.add(p as usize) += *ww0.add(k) * x;
                        *out.add(512 + p as usize) += *ww1.add(k) * x;
                    }
                }
            } else {
                // fe_fma in the reference's 128-bit abstraction is separate
                // multiply+add on x86, intentionally not fused here.
                for off in [0, 4] {
                    let dst0 = out.add(base as usize + off);
                    let dst1 = dst0.add(512);
                    _mm_storeu_ps(
                        dst0,
                        _mm_add_ps(
                            _mm_mul_ps(_mm_loadu_ps(ww0.add(off)), x),
                            _mm_loadu_ps(dst0),
                        ),
                    );
                    _mm_storeu_ps(
                        dst1,
                        _mm_add_ps(
                            _mm_mul_ps(_mm_loadu_ps(ww1.add(off)), x),
                            _mm_loadu_ps(dst1),
                        ),
                    );
                }
            }
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn run(
    model: &Model,
    s: &mut State,
    w: &mut Workspace,
    tier: SimdTier,
    dot: simd::Dot,
    frame: *mut f32,
) {
    let time = w.time.as_ptr() as *mut f32;
    let re = w.bins.as_ptr() as *mut f32;
    let im = re.add(513);
    let spec = w.spec.as_ptr() as *mut f32;
    let masked = w.masked.as_ptr() as *mut f32;
    let a = w.a.as_ptr() as *mut f32;
    let b = w.b.as_ptr() as *mut f32;
    let skip = w.skips.as_ptr() as *mut u16;
    ptr::copy_nonoverlapping(s.analysis.as_ptr(), time, 704);
    ptr::copy_nonoverlapping(frame, time.add(704), 320);
    ptr::copy_nonoverlapping(time.add(320), s.analysis.as_ptr() as *mut f32, 704);
    for n in (0..1024).step_by(8) {
        _mm256_storeu_ps(
            time.add(n),
            _mm256_mul_ps(
                _mm256_loadu_ps(time.add(n)),
                _mm256_loadu_ps(model.plan.window.as_ptr().add(n)),
            ),
        );
    }
    fft::forward(&model.plan, time, re, im, w.fft.as_ptr() as *mut f32);
    for f in (0..512).step_by(8) {
        let r = _mm256_loadu_ps(re.add(f));
        let i = _mm256_loadu_ps(im.add(f));
        let mag = _mm256_fmadd_ps(r, r, _mm256_mul_ps(i, i));
        let factor = simd::exp2(_mm256_mul_ps(
            _mm256_set1_ps(0.5 * (0.3_f32 - 1.)),
            simd::log2(_mm256_max_ps(mag, _mm256_set1_ps(1e-10))),
        ));
        _mm256_storeu_ps(spec.add(f), simd::normal(_mm256_mul_ps(r, factor)));
        _mm256_storeu_ps(spec.add(512 + f), simd::normal(_mm256_mul_ps(i, factor)));
    }
    encoder_pre(model, w, a);
    wino(w, tier, dot, &model.layers[0], a, b, skip.add(FC));
    wino(w, tier, dot, &model.layers[1], b, a, skip.add(2 * FC));
    wino(w, tier, dot, &model.layers[2], a, b, skip.add(3 * FC));
    linear(
        w,
        tier,
        dot,
        96,
        &model.layers[3].matrix,
        ptr::null(),
        skip.add(3 * FC),
        true,
        b,
        false,
        false,
    );
    linear(
        w,
        tier,
        dot,
        72,
        &model.layers[4].matrix,
        b,
        ptr::null(),
        true,
        a,
        false,
        false,
    );
    for block in 0..4 {
        gru(
            model,
            w,
            tier,
            dot,
            block,
            a,
            (s.hidden.as_ptr() as *mut u16).add(block * RF),
        );
        linear(
            w,
            tier,
            dot,
            72,
            &model.layers[7 + block * 5].matrix,
            w.gru_out.as_ptr(),
            ptr::null(),
            false,
            a,
            false,
            true,
        );
        if block == 0 {
            for i in (0..RF).step_by(8) {
                _mm256_storeu_ps(
                    a.add(i),
                    _mm256_add_ps(
                        _mm256_loadu_ps(a.add(i)),
                        _mm256_loadu_ps(model.pe.as_ptr().add(i)),
                    ),
                );
            }
        }
        attention(model, w, tier, dot, block, a);
    }
    linear(
        w,
        tier,
        dot,
        72,
        &model.layers[25].matrix,
        a,
        ptr::null(),
        true,
        b,
        false,
        false,
    );
    linear(
        w,
        tier,
        dot,
        128,
        &model.layers[26].matrix,
        b,
        ptr::null(),
        true,
        a,
        false,
        false,
    );
    for i in 0..3 {
        concat(
            w,
            tier,
            dot,
            &model.layers[27 + i * 2].matrix,
            a,
            skip.add((3 - i) * FC),
            b,
        );
        wino(
            w,
            tier,
            dot,
            &model.layers[28 + i * 2],
            b,
            a,
            ptr::null_mut(),
        );
    }
    concat(w, tier, dot, &model.layers[33].matrix, a, skip, b);
    upsample(model, b, time);
    for f in (0..512).step_by(8) {
        let sr = _mm256_loadu_ps(spec.add(f));
        let si = _mm256_loadu_ps(spec.add(512 + f));
        let mr = _mm256_loadu_ps(time.add(f));
        let mi = _mm256_loadu_ps(time.add(512 + f));
        _mm256_storeu_ps(
            masked.add(f),
            _mm256_fmsub_ps(sr, mr, _mm256_mul_ps(si, mi)),
        );
        _mm256_storeu_ps(
            masked.add(512 + f),
            _mm256_fmadd_ps(sr, mi, _mm256_mul_ps(si, mr)),
        );
    }
    for f in (0..512).step_by(8) {
        let r = _mm256_loadu_ps(masked.add(f));
        let i = _mm256_loadu_ps(masked.add(512 + f));
        let mag = _mm256_fmadd_ps(r, r, _mm256_mul_ps(i, i));
        let eps = _mm256_set1_ps(1e-20);
        let factor = simd::exp2(_mm256_mul_ps(
            _mm256_set1_ps(0.5 * (1. / 0.3_f32 - 1.)),
            simd::log2(_mm256_max_ps(mag, eps)),
        ));
        let factor = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(mag, eps), factor);
        _mm256_storeu_ps(re.add(f), simd::normal(_mm256_mul_ps(r, factor)));
        _mm256_storeu_ps(im.add(f), simd::normal(_mm256_mul_ps(i, factor)));
    }
    *re.add(512) = 0.;
    *im.add(512) = 0.;
    fft::inverse(&model.plan, re, im, time, w.fft.as_ptr() as *mut f32);
    for n in (0..1024).step_by(8) {
        let v = _mm256_loadu_ps(time.add(n));
        let win = _mm256_loadu_ps(model.plan.synthesis.as_ptr().add(n));
        let y = if n < 704 {
            _mm256_fmadd_ps(v, win, _mm256_loadu_ps(s.synthesis.as_ptr().add(n)))
        } else {
            _mm256_mul_ps(v, win)
        };
        _mm256_storeu_ps(time.add(n), simd::normal(y));
    }
    ptr::copy_nonoverlapping(time, frame, 320);
    ptr::copy_nonoverlapping(time.add(320), s.synthesis.as_ptr() as *mut f32, 704);
}
