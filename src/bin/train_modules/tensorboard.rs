use std::cell::RefCell;
use tensorboard_rs::summary_writer::SummaryWriter;

pub trait StatisticsCollector {
    fn add_image(&self, tag: &str, data: &[u8], dim: &[usize], step: usize);
    fn add_scalar(&self, tag: &str, value: f32, step: usize);
    fn flush(&self);
}

pub struct TensorboardStatisticsCollector {
    summary_writer: RefCell<SummaryWriter>,
}

impl TensorboardStatisticsCollector {
    pub fn new(path_to_log: &str) -> Self {
        Self {
            summary_writer: RefCell::new(SummaryWriter::new(path_to_log)),
        }
    }
}

impl StatisticsCollector for TensorboardStatisticsCollector {
    fn add_image(&self, tag: &str, data: &[u8], dim: &[usize], step: usize) {
        self.summary_writer
            .borrow_mut()
            .add_image(tag, data, dim, step);
    }

    fn add_scalar(&self, tag: &str, value: f32, step: usize) {
        self.summary_writer
            .borrow_mut()
            .add_scalar(tag, value, step);
    }

    fn flush(&self) {
        self.summary_writer.borrow_mut().flush();
    }
}
