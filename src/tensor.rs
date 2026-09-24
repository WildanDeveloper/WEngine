#[derive(Clone)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}
