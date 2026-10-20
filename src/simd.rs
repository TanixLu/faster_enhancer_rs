//! Private x86 microkernels. Callers establish the ISA once at initialization.
//! Pointer contracts: arrays cover the specified fixed graph dimensions; N is
//! and M are divisible by 8, K by 4, and input/output storage does not overlap unless
//! explicitly described. No pointer escapes any call.
#![allow(unsafe_op_in_unsafe_fn)]
// Fixed trip-count loops expose register tiles. Decimal coefficients are kept
// verbatim for comparison with the pinned C approximation polynomials.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::approx_constant
)]
use crate::{InitError, SimdTier};
use core::arch::x86_64::*;

pub(crate) fn select(requested: Option<SimdTier>) -> Result<SimdTier, InitError> {
    if !(is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c"))
    {
        return Err(InitError::UnsupportedCpu);
    }
    let vnni = is_x86_feature_detected!("avxvnni");
    let wide = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vnni");
    let tier = requested.unwrap_or(if wide {
        SimdTier::Avx512Vnni
    } else if vnni {
        SimdTier::AvxVnni
    } else {
        SimdTier::Avx2
    });
    if (tier == SimdTier::AvxVnni && !vnni) || (tier == SimdTier::Avx512Vnni && !wide) {
        return Err(InitError::UnsupportedCpu);
    }
    Ok(tier)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn sigmoid(x: __m256) -> __m256 {
    let x2 = _mm256_mul_ps(x, x);
    let mut num = _mm256_fmadd_ps(_mm256_set1_ps(0.00950985), x2, _mm256_set1_ps(6.02452230));
    num = _mm256_fmadd_ps(num, x2, _mm256_set1_ps(238.13200378));
    num = _mm256_mul_ps(num, x);
    let mut den = _mm256_fmadd_ps(_mm256_set1_ps(0.74287558), x2, _mm256_set1_ps(103.34200287));
    den = _mm256_fmadd_ps(den, x2, _mm256_set1_ps(952.72399902));
    let r = _mm256_fmadd_ps(num, _mm256_rcp_ps(den), _mm256_set1_ps(0.5));
    _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.), r))
}
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn tanh(x: __m256) -> __m256 {
    let x2 = _mm256_mul_ps(x, x);
    let mut num = _mm256_fmadd_ps(_mm256_set1_ps(0.60863042), x2, _mm256_set1_ps(96.39235687));
    num = _mm256_fmadd_ps(num, x2, _mm256_set1_ps(952.52801514));
    num = _mm256_mul_ps(num, x);
    let mut den = _mm256_fmadd_ps(
        _mm256_set1_ps(11.88600922),
        x2,
        _mm256_set1_ps(413.36801147),
    );
    den = _mm256_fmadd_ps(den, x2, _mm256_set1_ps(952.72399902));
    let r = _mm256_mul_ps(num, _mm256_rcp_ps(den));
    _mm256_max_ps(_mm256_set1_ps(-1.), _mm256_min_ps(_mm256_set1_ps(1.), r))
}
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn log2(x: __m256) -> __m256 {
    let ix = _mm256_castps_si256(x);
    let e = _mm256_sub_epi32(_mm256_srli_epi32::<23>(ix), _mm256_set1_epi32(127));
    let mb = _mm256_or_si256(
        _mm256_and_si256(ix, _mm256_set1_epi32(0x007fffff)),
        _mm256_set1_epi32(0x3f800000),
    );
    let y = _mm256_sub_ps(_mm256_castsi256_ps(mb), _mm256_set1_ps(1.));
    let mut p = _mm256_set1_ps(0.05500914);
    for c in [-0.20652259, 0.47917641, -0.71567517, 1.44256800] {
        p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(c));
    }
    _mm256_add_ps(_mm256_cvtepi32_ps(e), _mm256_mul_ps(y, p))
}
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp2(mut x: __m256) -> __m256 {
    x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-126.)),
        _mm256_set1_ps(127.),
    );
    let fk = _mm256_floor_ps(x);
    let k = _mm256_cvtps_epi32(fk);
    let r = _mm256_sub_ps(x, fk);
    let mut p = _mm256_set1_ps(0.0096180);
    for c in [0.0555041, 0.2402265, 0.6931472, 1.] {
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
    }
    let ks = _mm256_slli_epi32::<23>(_mm256_add_epi32(k, _mm256_set1_epi32(127)));
    _mm256_mul_ps(p, _mm256_castsi256_ps(ks))
}

/// Explicitly discard subnormal results without changing MXCSR. Integer masks
/// avoid denormal arithmetic in the cleanup itself. Normal values are unchanged.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn normal(x: __m256) -> __m256 {
    let bits = _mm256_castps_si256(x);
    let exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000));
    let tiny = _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256());
    let sign = _mm256_and_si256(bits, _mm256_set1_epi32(i32::MIN));
    _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_andnot_si256(tiny, bits),
        _mm256_and_si256(tiny, sign),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Quant {
    pub scale: f32,
    pub zp: i32,
}
pub(crate) fn qparams(mut min: f32, mut max: f32) -> Quant {
    min = min.min(0.);
    max = max.max(0.);
    let scale = (max - min).max(1e-12) / 255.;
    Quant {
        scale,
        zp: (-min / scale).round().clamp(0., 255.) as i32,
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pack8(q: __m256i, out: *mut i8) {
    let s = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q));
    let b = _mm_max_epi8(_mm_packs_epi16(s, s), _mm_set1_epi8(-127));
    _mm_storel_epi64(out.cast(), b);
}

#[target_feature(enable = "avx2")]
unsafe fn pack8_fp16(q: __m256i, out: *mut i8) {
    // The pinned C fe_fp16_quantize_asym uses saturating [-128,127], unlike
    // its fp32 and transposed-fp16 paths. Preserve the actual implementation.
    let s = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q));
    _mm_storel_epi64(out.cast(), _mm_packs_epi16(s, s));
}

/// Source may be fp32 or fp16, contiguous or transposed. Transpose is folded
/// into the quantization store; concat uses a shared qparam pair (see engine).
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn minmax(p: *const f32, ph: *const u16, n: usize) -> (f32, f32) {
    let mut lo = _mm256_set1_ps(f32::MAX);
    let mut hi = _mm256_set1_ps(-f32::MAX);
    for i in (0..n).step_by(8) {
        let v = if ph.is_null() {
            _mm256_loadu_ps(p.add(i))
        } else {
            _mm256_cvtph_ps(_mm_loadu_si128(ph.add(i).cast()))
        };
        lo = _mm256_min_ps(lo, v);
        hi = _mm256_max_ps(hi, v);
    }
    let mut a = [0.; 8];
    let mut b = [0.; 8];
    _mm256_storeu_ps(a.as_mut_ptr(), lo);
    _mm256_storeu_ps(b.as_mut_ptr(), hi);
    (
        a.into_iter().fold(f32::MAX, f32::min),
        b.into_iter().fold(-f32::MAX, f32::max),
    )
}
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn quantize(
    p: *const f32,
    ph: *const u16,
    m: usize,
    k: usize,
    transpose: bool,
    out: *mut i8,
    q: Quant,
) {
    let inv = _mm256_set1_ps(1. / q.scale);
    let off = _mm256_set1_ps((q.zp - 128) as f32);
    if !transpose {
        for i in (0..m * k).step_by(8) {
            let v = if ph.is_null() {
                _mm256_loadu_ps(p.add(i))
            } else {
                _mm256_cvtph_ps(_mm_loadu_si128(ph.add(i).cast()))
            };
            let qi = _mm256_cvtps_epi32(_mm256_fmadd_ps(v, inv, off));
            if ph.is_null() {
                pack8(qi, out.add(i));
            } else {
                pack8_fp16(qi, out.add(i));
            }
        }
    } else {
        // Four rows at a time: same 4x4 transpose + FMA/RNE as the C path.
        let vinv = _mm256_castps256_ps128(inv);
        let voff = _mm256_castps256_ps128(off);
        for r in (0..m).step_by(4) {
            for c in (0..k).step_by(4) {
                let mut v = [_mm_setzero_ps(); 4];
                for j in 0..4 {
                    let ix = (c + j) * m + r;
                    v[j] = if ph.is_null() {
                        _mm_loadu_ps(p.add(ix))
                    } else {
                        _mm_cvtph_ps(_mm_loadl_epi64(ph.add(ix).cast()))
                    };
                }
                let a = _mm_unpacklo_ps(v[0], v[1]);
                let b = _mm_unpackhi_ps(v[0], v[1]);
                let d = _mm_unpacklo_ps(v[2], v[3]);
                let e = _mm_unpackhi_ps(v[2], v[3]);
                let rows = [
                    _mm_movelh_ps(a, d),
                    _mm_movehl_ps(d, a),
                    _mm_movelh_ps(b, e),
                    _mm_movehl_ps(e, b),
                ];
                for j in 0..4 {
                    // The fp32 transpose path deliberately uses MUL+ADD;
                    // the fp16 transpose path uses FMA in the reference.
                    let scaled = if ph.is_null() {
                        _mm_add_ps(_mm_mul_ps(rows[j], vinv), voff)
                    } else {
                        _mm_fmadd_ps(rows[j], vinv, voff)
                    };
                    let qi = _mm_cvtps_epi32(scaled);
                    let s = _mm_packs_epi32(qi, qi);
                    let b = _mm_max_epi8(_mm_packs_epi16(s, s), _mm_set1_epi8(-127));
                    out.add((r + j) * k + c)
                        .cast::<i32>()
                        .write_unaligned(_mm_cvtsi128_si32(b));
                }
            }
        }
    }
}

pub(crate) struct Post {
    pub scale: *const f32,
    pub bias: *const f32,
    pub out: *mut f32,
    pub silu: bool,
    pub accumulate: bool,
}
pub(crate) type Dot =
    unsafe fn(usize, usize, usize, *const i8, *const i8, *const i16, *mut i32, *const Post);

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn store_acc(acc: __m256i, r: usize, n: usize, ld: usize, c: *mut i32, post: *const Post) {
    if post.is_null() {
        _mm256_storeu_si256(c.add(r * ld + n).cast(), acc);
    } else {
        let post = &*post;
        let v = _mm256_cvtepi32_ps(acc);
        let scale = _mm256_loadu_ps(post.scale.add(n));
        let mut y = if post.bias.is_null() {
            _mm256_mul_ps(v, scale)
        } else {
            _mm256_fmadd_ps(v, scale, _mm256_loadu_ps(post.bias.add(n)))
        };
        if post.silu {
            y = _mm256_mul_ps(y, sigmoid(y));
        }
        if post.accumulate {
            y = _mm256_add_ps(y, _mm256_loadu_ps(post.out.add(r * ld + n)));
        }
        _mm256_storeu_ps(post.out.add(r * ld + n), normal(y));
    }
}

/// K-pair weights are prepacked at Model initialization. Activations are
/// widened once into caller-owned workspace. The 8x8 integer tile is exact.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_avx2(
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    b: *const i8,
    wide: *const i16,
    c: *mut i32,
    post: *const Post,
) {
    let _ = a;
    for bn in (0..n).step_by(8) {
        for r in (0..m).step_by(8) {
            let rows = 8;
            let mut acc = [_mm256_setzero_si256(); 8];
            for kp in 0..k / 2 {
                let bv = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.add(bn * k + kp * 16).cast()));
                for j in 0..rows {
                    let av = _mm256_broadcastd_epi32(_mm_loadu_si32(
                        wide.add((r + j) * k + kp * 2).cast(),
                    ));
                    acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(av, bv));
                }
            }
            for j in 0..rows {
                store_acc(acc[j], r + j, bn, n, c, post);
            }
        }
    }
}

/// AVX-VNNI uses the VEX intrinsic, never AVX-512VL's EVEX vpdpbusd.
#[target_feature(enable = "avx2,fma,avxvnni")]
pub(crate) unsafe fn dot_vnni(
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    b: *const i8,
    _wide: *const i16,
    c: *mut i32,
    post: *const Post,
) {
    let flip = _mm256_set1_epi8(-128);
    let ones = _mm256_set1_epi8(1);
    for bn in (0..n).step_by(8) {
        let mut sum = _mm256_setzero_si256();
        for g in 0..k / 4 {
            sum = _mm256_dpbusd_avx_epi32(
                sum,
                ones,
                _mm256_loadu_si256(b.add(bn * k + g * 32).cast()),
            );
        }
        let correction = _mm256_slli_epi32::<7>(sum);
        for r in (0..m).step_by(8) {
            let rows = 8;
            let mut acc = [_mm256_setzero_si256(); 8];
            for g in 0..k / 4 {
                let bv = _mm256_loadu_si256(b.add(bn * k + g * 32).cast());
                for j in 0..rows {
                    let av = _mm256_xor_si256(
                        _mm256_broadcastd_epi32(_mm_loadu_si32(a.add((r + j) * k + g * 4).cast())),
                        flip,
                    );
                    acc[j] = _mm256_dpbusd_avx_epi32(acc[j], av, bv);
                }
            }
            for j in 0..rows {
                store_acc(_mm256_sub_epi32(acc[j], correction), r + j, bn, n, c, post);
            }
        }
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) unsafe fn dot_vnni512(
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    b: *const i8,
    _wide: *const i16,
    c: *mut i32,
    post: *const Post,
) {
    let flip = _mm512_set1_epi8(-128);
    let ones = _mm512_set1_epi8(1);
    for bn in (0..n).step_by(16) {
        let has_hi = bn + 8 < n;
        let mut sum = _mm512_setzero_si512();
        for g in 0..k / 4 {
            let lo = _mm256_loadu_si256(b.add(bn * k + g * 32).cast());
            let hi = if has_hi {
                _mm256_loadu_si256(b.add((bn + 8) * k + g * 32).cast())
            } else {
                _mm256_setzero_si256()
            };
            let bv = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
            sum = _mm512_dpbusd_epi32(sum, ones, bv);
        }
        let correction = _mm512_slli_epi32::<7>(sum);
        for r in (0..m).step_by(8) {
            let rows = 8;
            let mut acc = [_mm512_setzero_si512(); 8];
            for g in 0..k / 4 {
                let lo = _mm256_loadu_si256(b.add(bn * k + g * 32).cast());
                let hi = if has_hi {
                    _mm256_loadu_si256(b.add((bn + 8) * k + g * 32).cast())
                } else {
                    _mm256_setzero_si256()
                };
                let bv = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
                for j in 0..rows {
                    let word = a.add((r + j) * k + g * 4).cast::<i32>().read_unaligned();
                    let av = _mm512_xor_si512(_mm512_set1_epi32(word), flip);
                    acc[j] = _mm512_dpbusd_epi32(acc[j], av, bv);
                }
            }
            for j in 0..rows {
                let v = _mm512_sub_epi32(acc[j], correction);
                store_acc(_mm512_castsi512_si256(v), r + j, bn, n, c, post);
                if has_hi {
                    store_acc(_mm512_extracti64x4_epi64::<1>(v), r + j, bn + 8, n, c, post);
                }
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn expand(a: *const i8, out: *mut i16, n: usize) {
    let mut i = 0;
    while i + 16 <= n {
        _mm256_storeu_si256(
            out.add(i).cast(),
            _mm256_cvtepi8_epi16(_mm_loadu_si128(a.add(i).cast())),
        );
        i += 16;
    }
    while i < n {
        *out.add(i) = *a.add(i) as i16;
        i += 1;
    }
}

pub(crate) fn pair_pack(src: &[i8], dst: &mut [i8], n: usize, k: usize) {
    for bn in 0..n / 8 {
        for g in 0..k / 4 {
            for lane in 0..8 {
                for j in 0..4 {
                    dst[bn * k * 8 + (g * 2 + j / 2) * 16 + lane * 2 + j % 2] =
                        src[bn * k * 8 + g * 32 + lane * 4 + j];
                }
            }
        }
    }
}
