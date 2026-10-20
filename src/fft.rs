//! The reference's specialized 1024-real Stockham FFT: radix-4 x4 + radix-2.
//! All plan data belongs to Model and all temporary arrays to Workspace.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_range_loop)]
use crate::{InitError, MemoryRegion, memory::Buffer};
use core::arch::x86_64::*;

pub(crate) struct Plan {
    pub window: Buffer<f32>,
    pub synthesis: Buffer<f32>,
    twr: Buffer<f32>,
    twi: Buffer<f32>,
    r2r: Buffer<f32>,
    r2i: Buffer<f32>,
    realr: Buffer<f32>,
    reali: Buffer<f32>,
}
impl Plan {
    pub fn new() -> Result<Self, InitError> {
        let mut p = Self {
            window: Buffer::new(1024)?,
            synthesis: Buffer::new(1024)?,
            twr: Buffer::new(255)?,
            twi: Buffer::new(255)?,
            r2r: Buffer::new(256)?,
            r2i: Buffer::new(256)?,
            realr: Buffer::new(512)?,
            reali: Buffer::new(512)?,
        };
        let tau = 2. * std::f64::consts::PI;
        for n in 0..1024 {
            p.window[n] = (0.5 - 0.5 * (tau * n as f64 / 1024.).cos()) as f32;
        }
        let mut fold = Buffer::<f32>::new(4096)?;
        for f in 0..7 {
            for n in 0..1024 {
                fold[f * 320 + n] += p.window[n] * p.window[n];
            }
        }
        for n in 0..1024 {
            p.synthesis[n] = (p.window[n] / fold[960 + n]) * (1. / 1024.);
        }
        let mut off = 0;
        for l in [1, 4, 16, 64] {
            for q in 1..=3 {
                for k in 0..l {
                    let a = -tau * q as f64 * k as f64 / (l * 4) as f64;
                    p.twr[off + (q - 1) * l + k] = a.cos() as f32;
                    p.twi[off + (q - 1) * l + k] = a.sin() as f32;
                }
            }
            off += 3 * l;
        }
        for k in 0..256 {
            let a = -tau * k as f64 / 512.;
            p.r2r[k] = a.cos() as f32;
            p.r2i[k] = a.sin() as f32;
        }
        for k in 0..512 {
            let a = -tau * k as f64 / 1024.;
            p.realr[k] = a.cos() as f32;
            p.reali[k] = a.sin() as f32;
        }
        Ok(p)
    }
    pub fn regions(&self, out: &mut Vec<MemoryRegion>) {
        out.extend([
            self.window.region(),
            self.synthesis.region(),
            self.twr.region(),
            self.twi.region(),
            self.r2r.region(),
            self.r2i.region(),
            self.realr.region(),
            self.reali.region(),
        ]);
    }
    pub fn memory_size(&self) -> usize {
        (self.window.len()
            + self.synthesis.len()
            + self.twr.len()
            + self.twi.len()
            + self.r2r.len()
            + self.r2i.len()
            + self.realr.len()
            + self.reali.len())
            * size_of::<f32>()
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn cmul(ar: __m256, ai: __m256, br: __m256, bi: __m256, fused: bool) -> (__m256, __m256) {
    if fused {
        (
            _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi)),
            _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br)),
        )
    } else {
        (
            _mm256_sub_ps(_mm256_mul_ps(ar, br), _mm256_mul_ps(ai, bi)),
            _mm256_add_ps(_mm256_mul_ps(ar, bi), _mm256_mul_ps(ai, br)),
        )
    }
}
#[target_feature(enable = "avx2")]
unsafe fn butterfly(a: [(__m256, __m256); 4]) -> [(__m256, __m256); 4] {
    let t0 = (_mm256_add_ps(a[0].0, a[2].0), _mm256_add_ps(a[0].1, a[2].1));
    let t1 = (_mm256_sub_ps(a[0].0, a[2].0), _mm256_sub_ps(a[0].1, a[2].1));
    let t2 = (_mm256_add_ps(a[1].0, a[3].0), _mm256_add_ps(a[1].1, a[3].1));
    let t3 = (_mm256_sub_ps(a[1].0, a[3].0), _mm256_sub_ps(a[1].1, a[3].1));
    [
        (_mm256_add_ps(t0.0, t2.0), _mm256_add_ps(t0.1, t2.1)),
        (_mm256_add_ps(t1.0, t3.1), _mm256_sub_ps(t1.1, t3.0)),
        (_mm256_sub_ps(t0.0, t2.0), _mm256_sub_ps(t0.1, t2.1)),
        (_mm256_sub_ps(t1.0, t3.1), _mm256_add_ps(t1.1, t3.0)),
    ]
}
#[target_feature(enable = "avx2")]
unsafe fn transpose_store(out: *mut f32, o: [__m256; 4]) {
    let t0 = _mm256_unpacklo_ps(o[0], o[1]);
    let t1 = _mm256_unpackhi_ps(o[0], o[1]);
    let t2 = _mm256_unpacklo_ps(o[2], o[3]);
    let t3 = _mm256_unpackhi_ps(o[2], o[3]);
    let r0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let r1 = _mm256_shuffle_ps::<0xee>(t0, t2);
    let r2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let r3 = _mm256_shuffle_ps::<0xee>(t1, t3);
    _mm256_storeu_ps(out, _mm256_permute2f128_ps::<0x20>(r0, r1));
    _mm256_storeu_ps(out.add(8), _mm256_permute2f128_ps::<0x20>(r2, r3));
    _mm256_storeu_ps(out.add(16), _mm256_permute2f128_ps::<0x31>(r0, r1));
    _mm256_storeu_ps(out.add(24), _mm256_permute2f128_ps::<0x31>(r2, r3));
}
#[target_feature(enable = "avx2")]
unsafe fn load(p: *const f32, small: bool) -> __m256 {
    if small {
        _mm256_insertf128_ps::<0>(_mm256_setzero_ps(), _mm_loadu_ps(p))
    } else {
        _mm256_loadu_ps(p)
    }
}
#[target_feature(enable = "avx2")]
unsafe fn store(p: *mut f32, v: __m256, small: bool) {
    if small {
        _mm_storeu_ps(p, _mm256_castps256_ps128(v));
    } else {
        _mm256_storeu_ps(p, v);
    }
}

/// scratch covers 2048 floats; all four arrays are disjoint. Input and output
/// cover 512 complex elements and do not alias scratch.
#[target_feature(enable = "avx2,fma")]
unsafe fn complex(
    p: &Plan,
    ir: *const f32,
    ii: *const f32,
    or: *mut f32,
    oi: *mut f32,
    scratch: *mut f32,
) {
    let br = [scratch, scratch.add(512)];
    let bi = [scratch.add(1024), scratch.add(1536)];
    for j in (0..128).step_by(8) {
        let mut a = [(_mm256_setzero_ps(), _mm256_setzero_ps()); 4];
        for q in 0..4 {
            a[q] = (
                _mm256_loadu_ps(ir.add(j + q * 128)),
                _mm256_loadu_ps(ii.add(j + q * 128)),
            );
        }
        let o = butterfly(a);
        transpose_store(br[0].add(j * 4), [o[0].0, o[1].0, o[2].0, o[3].0]);
        transpose_store(bi[0].add(j * 4), [o[0].1, o[1].1, o[2].1, o[3].1]);
    }
    let mut parity = 0;
    let mut off = 3;
    for l in [4, 16, 64] {
        let small = l == 4;
        for j in 0..512 / (4 * l) {
            for k in (0..l).step_by(if small { 4 } else { 8 }) {
                let base = j * l + k;
                let mut a = [(_mm256_setzero_ps(), _mm256_setzero_ps()); 4];
                for q in 0..4 {
                    a[q] = (
                        load(br[parity].add(base + q * 128), small),
                        load(bi[parity].add(base + q * 128), small),
                    );
                    if q > 0 {
                        a[q] = cmul(
                            a[q].0,
                            a[q].1,
                            load(p.twr.as_ptr().add(off + (q - 1) * l + k), small),
                            load(p.twi.as_ptr().add(off + (q - 1) * l + k), small),
                            !small,
                        );
                    }
                }
                let o = butterfly(a);
                for q in 0..4 {
                    let b = j * 4 * l + k + q * l;
                    store(br[1 - parity].add(b), o[q].0, small);
                    store(bi[1 - parity].add(b), o[q].1, small);
                }
            }
        }
        parity = 1 - parity;
        off += 3 * l;
    }
    for k in (0..256).step_by(8) {
        let ar = _mm256_loadu_ps(br[parity].add(k));
        let ai = _mm256_loadu_ps(bi[parity].add(k));
        let (rr, ri) = cmul(
            _mm256_loadu_ps(br[parity].add(k + 256)),
            _mm256_loadu_ps(bi[parity].add(k + 256)),
            _mm256_loadu_ps(p.r2r.as_ptr().add(k)),
            _mm256_loadu_ps(p.r2i.as_ptr().add(k)),
            true,
        );
        _mm256_storeu_ps(or.add(k), _mm256_add_ps(ar, rr));
        _mm256_storeu_ps(oi.add(k), _mm256_add_ps(ai, ri));
        _mm256_storeu_ps(or.add(k + 256), _mm256_sub_ps(ar, rr));
        _mm256_storeu_ps(oi.add(k + 256), _mm256_sub_ps(ai, ri));
    }
}

/// scratch covers 4096 floats. re/im cover 513 floats each.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn forward(
    p: &Plan,
    input: *const f32,
    re: *mut f32,
    im: *mut f32,
    scratch: *mut f32,
) {
    let pr = scratch;
    let pi = scratch.add(512);
    let zr = scratch.add(1024);
    let zi = scratch.add(1536);
    for k in (0..512).step_by(8) {
        let v0 = _mm256_loadu_ps(input.add(2 * k));
        let v1 = _mm256_loadu_ps(input.add(2 * k + 8));
        let ev = _mm256_shuffle_ps::<0x88>(v0, v1);
        let od = _mm256_shuffle_ps::<0xdd>(v0, v1);
        _mm256_storeu_ps(
            pr.add(k),
            _mm256_castpd_ps(_mm256_permute4x64_pd::<0xd8>(_mm256_castps_pd(ev))),
        );
        _mm256_storeu_ps(
            pi.add(k),
            _mm256_castpd_ps(_mm256_permute4x64_pd::<0xd8>(_mm256_castps_pd(od))),
        );
    }
    complex(p, pr, pi, zr, zi, scratch.add(2048));
    *re = *zr + *zi;
    *im = 0.;
    *re.add(512) = *zr - *zi;
    *im.add(512) = 0.;
    let half = _mm256_set1_ps(0.5);
    let reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let mut k = 1;
    while k + 8 <= 512 {
        let ar = _mm256_loadu_ps(zr.add(k));
        let ai = _mm256_loadu_ps(zi.add(k));
        let nr = _mm256_permutevar8x32_ps(_mm256_loadu_ps(zr.add(505 - k)), reverse);
        let ni = _mm256_permutevar8x32_ps(_mm256_loadu_ps(zi.add(505 - k)), reverse);
        let sr = _mm256_mul_ps(half, _mm256_add_ps(ar, nr));
        let si = _mm256_mul_ps(half, _mm256_sub_ps(ai, ni));
        let dr = _mm256_mul_ps(half, _mm256_sub_ps(ar, nr));
        let di = _mm256_mul_ps(half, _mm256_add_ps(ai, ni));
        let (rr, ri) = cmul(
            dr,
            di,
            _mm256_loadu_ps(p.realr.as_ptr().add(k)),
            _mm256_loadu_ps(p.reali.as_ptr().add(k)),
            true,
        );
        _mm256_storeu_ps(re.add(k), _mm256_add_ps(sr, ri));
        _mm256_storeu_ps(im.add(k), _mm256_sub_ps(si, rr));
        k += 8;
    }
    while k < 512 {
        let (ar, ai, nr, ni) = (*zr.add(k), *zi.add(k), *zr.add(512 - k), *zi.add(512 - k));
        let (sr, si, dr, di) = (
            0.5 * (ar + nr),
            0.5 * (ai - ni),
            0.5 * (ar - nr),
            0.5 * (ai + ni),
        );
        let (wr, wi) = (p.realr[k], p.reali[k]);
        *re.add(k) = sr + (dr * wi + di * wr);
        *im.add(k) = si - (dr * wr - di * wi);
        k += 1;
    }
}
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn inverse(
    p: &Plan,
    re: *const f32,
    im: *const f32,
    out: *mut f32,
    scratch: *mut f32,
) {
    let zr = scratch;
    let zi = scratch.add(512);
    let wr = scratch.add(1024);
    let wi = scratch.add(1536);
    *zr = *re + *re.add(512);
    *zi = *re - *re.add(512);
    let reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let zero = _mm256_setzero_ps();
    let mut k = 1;
    while k + 8 <= 512 {
        let ar = _mm256_loadu_ps(re.add(k));
        let ai = _mm256_loadu_ps(im.add(k));
        let nr = _mm256_permutevar8x32_ps(_mm256_loadu_ps(re.add(505 - k)), reverse);
        let ni = _mm256_permutevar8x32_ps(_mm256_loadu_ps(im.add(505 - k)), reverse);
        let sr = _mm256_add_ps(ar, nr);
        let si = _mm256_sub_ps(ai, ni);
        let dr = _mm256_sub_ps(ar, nr);
        let di = _mm256_add_ps(ai, ni);
        let (rr, ri) = cmul(
            dr,
            di,
            _mm256_loadu_ps(p.realr.as_ptr().add(k)),
            _mm256_sub_ps(zero, _mm256_loadu_ps(p.reali.as_ptr().add(k))),
            true,
        );
        _mm256_storeu_ps(zr.add(k), _mm256_sub_ps(sr, ri));
        _mm256_storeu_ps(zi.add(k), _mm256_add_ps(si, rr));
        k += 8;
    }
    while k < 512 {
        let (ar, ai, nr, ni) = (*re.add(k), *im.add(k), *re.add(512 - k), *im.add(512 - k));
        let (sr, si, dr, di) = (ar + nr, ai - ni, ar - nr, ai + ni);
        let (tr, ti) = (p.realr[k], -p.reali[k]);
        *zr.add(k) = sr - (dr * ti + di * tr);
        *zi.add(k) = si + (dr * tr - di * ti);
        k += 1;
    }
    for k in (0..512).step_by(8) {
        _mm256_storeu_ps(zi.add(k), _mm256_sub_ps(zero, _mm256_loadu_ps(zi.add(k))));
    }
    complex(p, zr, zi, wr, wi, scratch.add(2048));
    for k in (0..512).step_by(8) {
        let r = _mm256_loadu_ps(wr.add(k));
        let i = _mm256_sub_ps(zero, _mm256_loadu_ps(wi.add(k)));
        let lo = _mm256_unpacklo_ps(r, i);
        let hi = _mm256_unpackhi_ps(r, i);
        _mm256_storeu_ps(out.add(2 * k), _mm256_permute2f128_ps::<0x20>(lo, hi));
        _mm256_storeu_ps(out.add(2 * k + 8), _mm256_permute2f128_ps::<0x31>(lo, hi));
    }
}
