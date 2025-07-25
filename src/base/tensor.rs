#![allow(dead_code)] // Suppress "is never used" warnings globally in this file

#[derive(Clone)]
struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides: Vec<usize> = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn prod(dims: &[usize]) -> usize {
    dims.iter().copied().product()
}

impl Tensor {
    pub fn new<T>(shape: &[usize], values: &[T]) -> Self
    where
        T: Into<f64> + Copy,
    {
        if prod(shape) != values.len() {
            panic!("Shape Mismatch");
        }
        let mut data: Vec<f64> = Vec::with_capacity(values.len());
        for &v in values {
            data.push(v.into());
        }
        let strides: Vec<usize> = compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn new_from_vec_1d<T>(shape: &[usize], data: &Vec<T>) -> Self
    where
        T: Into<f64> + Copy,
    {
        if prod(shape) != data.len() {
            panic!("Shape Mismatch");
        }
        let data: Vec<f64> = data.iter().map(|&v| v.into()).collect();
        let strides: Vec<usize> = compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn reshape(&mut self, new_shape: &[usize]) {
        if prod(new_shape) != self.data.len() {
            panic!("Trying to reshape to invalid shape");
        }
        self.shape = new_shape.to_vec();
        self.strides = compute_strides(new_shape);
    }
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

    #[test]
    fn test_compute_strides() {
        let mut shape: Vec<usize> = vec![2];
        let mut strides: Vec<usize> = compute_strides(&shape);
        assert_eq!(strides, vec!(1));
        shape.push(3);
        strides = compute_strides(&shape);
        assert_eq!(strides, vec!(3, 1));
        shape.push(5);
        strides = compute_strides(&shape);
        assert_eq!(strides, vec!(15, 5, 1));
    }

    #[test]
    fn test_prod() {
        let shape: [usize; 1] = [2];
        let strides: usize = prod(&shape);
        assert_eq!(strides, 2);
        let shape: [usize; 2] = [2, 3];
        let strides: usize = prod(&shape);
        assert_eq!(strides, 6);
        let shape: [usize; 3] = [2, 3, 5];
        let strides: usize = prod(&shape);
        assert_eq!(strides, 30);
    }

    #[test]
    fn test_new_tensor_init() {
        let tensor: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 4]);
        assert_eq!(tensor.data, vec![1., 2., 3., 4.]);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.strides, vec![2, 1]);
    }

    #[test]
    #[should_panic]
    fn test_new_tensor_init_should_panic() {
        let tensor: Tensor = Tensor::new(&[2, 3], &[1, 2, 3, 4]);
        assert_eq!(tensor.data, vec![1., 2., 3., 4.]);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.strides, vec![2, 1]);
    }

    #[test]
    #[should_panic]
    fn test_new_tensor_init_from_vec_1d() {
        let tensor: Tensor = Tensor::new_from_vec_1d(&[2, 3], &vec![1, 2, 3, 4]);
        assert_eq!(tensor.data, vec![1., 2., 3., 4.]);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.strides, vec![2, 1]);
    }

    #[test]
    fn test_reshape() {
        let mut tensor: Tensor = Tensor::new_from_vec_1d(&[2, 3], &vec![1, 2, 3, 4, 5, 6]);
        tensor.reshape(&[3, 2]);
        assert_eq!(tensor.shape, [3, 2]);
        assert_eq!(tensor.strides, [2, 1]);
    }
}
