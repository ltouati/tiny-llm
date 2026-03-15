use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use tiny_llm::{Config, TinyLLM};
use tiny_llm::kernel::fused_cross_entropy::FusedCrossEntropy;
use tiny_llm::kernel::fused_adamw::{FusedAdamW, ParamsAdamW};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let config = Config { vocab_size: 50257, hidden_dim: 768, ffn_dim: 3072, num_layers: 12, num_heads: 12, num_kv_heads: 4, seq_len: 128 };
    let mut varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::BF16, &device);
    let model = TinyLLM::new(config.clone(), vb)?;
    
    let params = ParamsAdamW { lr: 0.0001, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.1 };
    let mut optim = FusedAdamW::new(varmap.all_vars(), params)?;
    
    let x = Tensor::zeros((2, 128), candle_core::DType::U32, &device)?;
    let y = Tensor::zeros((2, 128), candle_core::DType::U32, &device)?;
    
    for _ in 0..2 {
        let (logits, _) = model.forward(&x, 0)?;
        let targets = y.flatten_all()?;
        let cross_entropy = FusedCrossEntropy { targets };
        let unscaled_losses = logits.flatten_to(1)?.apply_op1(cross_entropy)?;
        let loss = unscaled_losses.mean_all()?;
        
        let scaled_loss = (loss / 2.0)?;
        let grads = scaled_loss.backward()?;
        optim.accumulate(&grads, 1)?;
        model.clear_kv_cache();
    }
    Ok(())
}
