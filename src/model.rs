use crate::{InitError, MemoryRegion, SimdTier, fft::Plan, memory::Buffer, simd};

static EMBEDDED_WEIGHTS: &[u8] = include_bytes!("../weights/fe.q8");

pub(crate) struct Matrix {
    pub n: usize,
    pub k: usize,
    pub q: Buffer<i8>,
    pub pair: Buffer<i8>,
    pub scales: Buffer<f32>,
    pub sums: Buffer<i32>,
    pub bias: Buffer<f32>,
    pub has_bias: bool,
}
impl Matrix {
    fn new(n: usize, k: usize, has_bias: bool) -> Result<Self, InitError> {
        Ok(Self {
            n,
            k,
            q: Buffer::new(n * k)?,
            pair: Buffer::new(n * k)?,
            scales: Buffer::new(n)?,
            sums: Buffer::new(n)?,
            bias: Buffer::new(n)?,
            has_bias,
        })
    }
    fn prepare(&mut self) {
        for n in 0..self.n {
            self.sums[n] = (0..self.k)
                .map(|k| self.q[index(n, k, self.k)] as i32)
                .sum();
        }
        simd::pair_pack(&self.q, &mut self.pair, self.n, self.k);
    }
    pub fn regions(&self, out: &mut Vec<MemoryRegion>) {
        out.extend([
            self.q.region(),
            self.pair.region(),
            self.scales.region(),
            self.sums.region(),
            self.bias.region(),
        ]);
    }
    fn memory_size(&self) -> usize {
        self.q.region().len
            + self.pair.region().len
            + self.scales.region().len
            + self.sums.region().len
            + self.bias.region().len
    }
}
pub(crate) fn index(n: usize, k: usize, cols: usize) -> usize {
    n / 8 * cols * 8 + k / 4 * 32 + n % 8 * 4 + k % 4
}

pub(crate) struct Layer {
    pub matrix: Matrix,
    pub wino: Option<[Matrix; 4]>,
}

/// Validated, owning, immutable FastEnhancer-Medium weights and derived plans.
/// It retains no references to the input blob and may be shared with `Arc`.
pub struct Model {
    pub(crate) layers: Box<[Layer; 34]>,
    pub(crate) enc_weight: Buffer<f32>,
    pub(crate) enc_bias: Buffer<f32>,
    pub(crate) up_weight: Buffer<f32>,
    pub(crate) up_bias: Buffer<f32>,
    pub(crate) pe: Buffer<f32>,
    pub(crate) combined_bias: Buffer<f32>,
    pub(crate) plan: Plan,
}
impl Model {
    /// Load the FastEnhancer-Medium weights embedded in the library.
    pub fn new() -> Result<Self, InitError> {
        Self::from_weights(EMBEDDED_WEIGHTS)
    }

    /// Load external weights in the supported FM_W8_03 version 1 format.
    pub fn from_weights(weights: &[u8]) -> Result<Self, InitError> {
        simd::select(Some(SimdTier::Avx2))?;
        if weights.len() != 565_108 {
            return Err(InitError::InvalidLength);
        }
        if &weights[..8] != b"FM_W8_03" {
            return Err(InitError::InvalidMagic);
        }
        if u32::from_le_bytes(
            weights[8..12]
                .try_into()
                .map_err(|_| InitError::InvalidLength)?,
        ) != 1
        {
            return Err(InitError::UnsupportedVersion);
        }
        let mut r = Reader {
            bytes: weights,
            at: 12,
        };
        let enc_src = r.floats(96 * 16, false)?;
        let enc_bias = r.floats(96, false)?;
        let mut enc_weight = Buffer::new(96 * 16)?;
        // SGEMM NR=8 layout [N/8,K,8].
        for n in 0..96 {
            for k in 0..16 {
                enc_weight[n / 8 * 16 * 8 + k * 8 + n % 8] = enc_src[n * 16 + k];
            }
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(34)
            .map_err(|_| InitError::Allocation)?;
        for _ in 0..3 {
            layers.push(r.layer(96, 288, true, true)?);
        }
        layers.push(r.layer(72, 128, false, false)?);
        layers.push(r.layer(72, 96, true, false)?);
        let mut pe = Buffer::new(72 * 72)?;
        let mut combined_bias = Buffer::new(4 * 216)?;
        for b in 0..4 {
            let ih = r.layer(216, 72, true, false)?;
            let hh = r.layer(216, 72, true, false)?;
            for j in 0..216 {
                combined_bias[b * 216 + j] = ih.matrix.bias[j] + hh.matrix.bias[j];
            }
            layers.push(ih);
            layers.push(hh);
            layers.push(r.layer(72, 72, true, false)?);
            layers.push(r.layer(216, 72, false, false)?);
            layers.push(r.layer(72, 72, true, false)?);
            if b == 0 {
                pe = r.floats(72 * 72, false)?;
            }
        }
        layers.push(r.layer(128, 72, false, false)?);
        layers.push(r.layer(96, 72, true, false)?);
        for _ in 0..3 {
            layers.push(r.layer(96, 192, true, false)?);
            layers.push(r.layer(96, 288, true, true)?);
        }
        layers.push(r.layer(96, 192, true, false)?);
        let up_weight = r.floats(96 * 2 * 8, false)?;
        let up_bias = r.floats(2, false)?;
        if r.at != weights.len() {
            return Err(InitError::InvalidLength);
        }
        let layers = layers
            .into_boxed_slice()
            .try_into()
            .map_err(|_| InitError::InvalidLength)?;
        Ok(Self {
            layers,
            enc_weight,
            enc_bias,
            up_weight,
            up_bias,
            pe,
            combined_bias,
            plan: Plan::new()?,
        })
    }

    /// All large model allocations. Calling this query allocates a small Vec;
    /// call it off the processing thread after placing the model at its final
    /// address (normally in Arc). Numeric allocations are 64-byte aligned.
    pub fn memory_regions(&self) -> Vec<MemoryRegion> {
        let mut out = Vec::new();
        out.push(MemoryRegion {
            address: (self as *const Self).cast(),
            len: size_of::<Self>(),
        });
        out.push(MemoryRegion {
            address: self.layers.as_ptr().cast(),
            len: size_of::<[Layer; 34]>(),
        });
        out.extend([
            self.enc_weight.region(),
            self.enc_bias.region(),
            self.up_weight.region(),
            self.up_bias.region(),
            self.pe.region(),
            self.combined_bias.region(),
        ]);
        self.plan.regions(&mut out);
        for l in self.layers.iter() {
            l.matrix.regions(&mut out);
            if let Some(w) = &l.wino {
                for m in w {
                    m.regions(&mut out);
                }
            }
        }
        out
    }
    pub fn memory_size(&self) -> usize {
        let mut bytes = size_of::<Self>() + size_of::<[Layer; 34]>() + self.plan.memory_size();
        bytes += [
            self.enc_weight.region(),
            self.enc_bias.region(),
            self.up_weight.region(),
            self.up_bias.region(),
            self.pe.region(),
            self.combined_bias.region(),
        ]
        .iter()
        .map(|r| r.len)
        .sum::<usize>();
        for layer in self.layers.iter() {
            bytes += layer.matrix.memory_size();
            if let Some(wino) = &layer.wino {
                for matrix in wino {
                    bytes += matrix.memory_size();
                }
            }
        }
        bytes
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl Reader<'_> {
    fn floats(&mut self, n: usize, scale: bool) -> Result<Buffer<f32>, InitError> {
        let end = self.at.checked_add(n * 4).ok_or(InitError::InvalidLength)?;
        let bytes = self
            .bytes
            .get(self.at..end)
            .ok_or(InitError::InvalidLength)?;
        let mut values = Buffer::new(n)?;
        for (i, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            // Fixed pretrained graph: reject NaN, infinity, subnormal scales,
            // and unreasonable magnitudes before they can reach any kernel.
            if !x.is_finite() || x.abs() > 1e6 || (scale && (!x.is_normal() || x <= 0.)) {
                return Err(InitError::InvalidWeight);
            }
            values[i] = x;
        }
        self.at = end;
        Ok(values)
    }
    fn layer(&mut self, n: usize, k: usize, bias: bool, wino: bool) -> Result<Layer, InitError> {
        let end = self.at.checked_add(n * k).ok_or(InitError::InvalidLength)?;
        let bytes = self
            .bytes
            .get(self.at..end)
            .ok_or(InitError::InvalidLength)?;
        let mut matrix = Matrix::new(n, k, bias)?;
        for (x, &b) in matrix.q.iter_mut().zip(bytes) {
            if b == 128 {
                return Err(InitError::InvalidWeight);
            }
            *x = b as i8;
        }
        self.at = end;
        matrix.scales = self.floats(n, true)?;
        if bias {
            matrix.bias = self.floats(n, false)?;
        }
        matrix.prepare();
        let wino = if wino { Some(derive(&matrix)?) } else { None };
        Ok(Layer { matrix, wino })
    }
}

fn derive(src: &Matrix) -> Result<[Matrix; 4], InitError> {
    let mut out = [
        Matrix::new(96, 96, true)?,
        Matrix::new(96, 96, true)?,
        Matrix::new(96, 96, true)?,
        Matrix::new(96, 96, true)?,
    ];
    let mut transformed = Buffer::<f32>::new(4 * 96 * 96)?;
    for n in 0..96 {
        for c in 0..96 {
            let w0 = src.scales[n] * src.q[index(n, c * 3, 288)] as f32;
            let w1 = src.scales[n] * src.q[index(n, c * 3 + 1, 288)] as f32;
            let w2 = src.scales[n] * src.q[index(n, c * 3 + 2, 288)] as f32;
            for (i, x) in [w0, 0.5 * ((w0 + w1) + w2), 0.5 * ((w0 - w1) + w2), w2]
                .into_iter()
                .enumerate()
            {
                transformed[i * 96 * 96 + n * 96 + c] = x;
            }
        }
    }
    for (i, m) in out.iter_mut().enumerate() {
        for n in 0..96 {
            let row = &transformed[i * 96 * 96 + n * 96..i * 96 * 96 + (n + 1) * 96];
            let max = row.iter().map(|x| x.abs()).fold(0., f32::max);
            let scale = if max == 0. { 1. } else { max / 127. };
            if !scale.is_normal() {
                return Err(InitError::InvalidWeight);
            }
            m.scales[n] = scale;
            for (c, &value) in row.iter().enumerate() {
                m.q[index(n, c, 96)] = (value / scale).round().clamp(-127., 127.) as i8;
            }
        }
        m.prepare();
    }
    Ok(out)
}
