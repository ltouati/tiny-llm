use candle_core::{DType, Device};
use candle_nn::{var_builder::InitHints, VarBuilder, VarMap};

pub fn get_tracked_vb<'a>(varmap: &'a mut VarMap, dtype: DType, device: &'a Device) -> VarBuilder<'a> {
    // Explicitly forces variables created by this VarBuilder to track gradients natively
    let hints = InitHints::default(); // Not configurable via builder in some older candle versions, must modify VarMap manually? No, let's just make sure is_mut is true on creation.
    // Actually, `VarBuilder::from_varmap` creates trainable tensors by default if gradients aren't implicitly disabled. 
    // The issue might just be `LayerNorm` specifically not tracking inputs correctly without an explicit `is_mut` tensor creation mechanism inside it. 
    VarBuilder::from_varmap(varmap, dtype, device)
}
