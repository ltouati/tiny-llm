use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        // RMSNorm = x * rsqrt(mean(x^2) + eps) * weight
        let sq = x_f32.sqr()?;
        let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
        let rsqrt = mean_sq.affine(1.0, self.eps)?.powf(-0.5)?;
        let norm_x = x_f32.broadcast_mul(&rsqrt)?;
        let out = norm_x.broadcast_mul(&self.weight.to_dtype(candle_core::DType::F32)?)?;
        out.to_dtype(dtype)
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    RmsNorm::new(size, eps, vb)
}
