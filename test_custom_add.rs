use candle_core::{CustomOp2, Result, Tensor};

struct StrictAdd;
impl CustomOp2 for StrictAdd {
    fn name(&self) -> &'static str { "strict_add" }
    fn bwd(&self, _arg1: &Tensor, _arg2: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((Some(grad_res.clone()), Some(grad_res.clone())))
    }
    fn fwd(&self, arg1: &Tensor, arg2: &Tensor) -> Result<Tensor> {
        arg1.broadcast_add(arg2)
    }
}
fn main() {}
