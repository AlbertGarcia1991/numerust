#[derive(Clone)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor() {
        let data: Vec<f64> = vec![1., 2., 3., 4.];
        let shape: Vec<usize> = vec![2, 2];
        let strides: Vec<usize> = vec![2, 1];
        let tensor: Tensor = Tensor {
            data: data.clone(),
            shape: shape.clone(),
            strides: strides.clone(),
        };
        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.strides, strides);
    }
}
