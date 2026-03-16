use burn::module::Module;
use burn::module::Param;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for `RMSNorm`.
#[derive(burn::config::Config, Debug)]
pub struct RMSNormConfig {
    /// The size of the expected features.
    pub d_model: usize,
    /// A value added to the denominator for numerical stability. Default: `1e-5`.
    #[config(default = "1e-5")]
    pub epsilon: f64,
}

/// Applies Root Mean Square Normalization over a tensor.
#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl RMSNormConfig {
    /// Initialize a new `RMSNorm` module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RMSNorm<B> {
        let weight = Tensor::<B, 1>::ones([self.d_model], device);
        RMSNorm {
            weight: Param::from_tensor(weight),
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> RMSNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - `x`: `[..., d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // Calculate the variance element-wise: mean(x^2)
        let var = x.clone().powf_scalar(2.0).mean_dim(D - 1);

        // norm = x / sqrt(var + eps)
        let norm = x.div(var.add_scalar(self.epsilon).sqrt());

        // Scale by learned weight
        norm.mul(self.weight.val().unsqueeze())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray;

    #[test]
    fn test_rmsnorm_forward() {
        let device = Default::default();
        let config = RMSNormConfig::new(768);
        let norm = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random([2, 16, 768], Distribution::Normal(0.0, 1.0), &device);
        let output = norm.forward(x);

        assert_eq!(output.dims(), [2, 16, 768]);
    }
}
