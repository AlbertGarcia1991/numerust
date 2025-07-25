#![allow(dead_code)] // Suppress "is never used" warnings globally in this file

use std::ops::{Add, Div, Index, IndexMut, Mul, Rem, Sub};

#[derive(Clone, Debug)]
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

    pub fn dims(&self) -> usize {
        self.shape.len()
    }

    // TODO: Test return Box to remove overheading of having to unwrap result. Benchmark speed and if Box might be implemented somewhere else
    pub fn at(&self, indices: &[usize]) -> Result<Tensor, f64> {
        let mut flat_index: usize = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                panic!("Index out of bounds for dimension {}", i);
            }
            flat_index += idx * self.strides[i];
        }
        // If all dimensions are specified, return the single value
        if indices.len() == self.shape.len() {
            return Err(self.data[flat_index]);
        }
        // Otherwise, calculate the sub-tensor
        let sub_shape: Vec<usize> = self.shape[indices.len()..].to_vec();
        let sub_strides: Vec<usize> = self.strides[indices.len()..].to_vec();
        let sub_size: usize = prod(&sub_shape);
        let sub_data: Vec<f64> = self.data[flat_index..flat_index + sub_size].to_vec();
        Ok(Tensor {
            data: sub_data,
            shape: sub_shape,
            strides: sub_strides,
        })
    }
}

impl Index<usize> for Tensor {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}

macro_rules! impl_binary_tensor_op {
    ($trait:ident, $func:ident) => {
        impl $trait for &Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Shape mismatch");
                let data = self
                    .data
                    .iter()
                    .zip(&rhs.data)
                    .map(|(a, b)| a.$func(*b))
                    .collect();
                Tensor {
                    data,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
        }
        impl $trait<f64> for &Tensor {
            type Output = Tensor;
            fn $func(self, rhs: f64) -> Self::Output {
                let data = self.data.iter().map(|a| a.$func(rhs)).collect();
                Tensor {
                    data,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
        }
    };
}

impl_binary_tensor_op!(Add, add);
impl_binary_tensor_op!(Sub, sub);
impl_binary_tensor_op!(Mul, mul);
impl_binary_tensor_op!(Div, div);
impl_binary_tensor_op!(Rem, rem);

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
        assert_eq!(tensor.dims(), 2);
    }

    #[test]
    fn test_reshape() {
        let mut tensor: Tensor = Tensor::new_from_vec_1d(&[2, 3], &vec![1, 2, 3, 4, 5, 6]);
        tensor.reshape(&[3, 2]);
        assert_eq!(tensor.shape, [3, 2]);
        assert_eq!(tensor.strides, [2, 1]);
    }

    #[test]
    fn test_reshape_from_1d() {
        let mut tensor: Tensor = Tensor::new_from_vec_1d(&[1, 6], &vec![1, 2, 3, 4, 5, 6]);
        tensor.reshape(&[3, 2]);
        assert_eq!(tensor.shape, [3, 2]);
        assert_eq!(tensor.strides, [2, 1]);
    }

    #[test]
    fn test_reshape_to_1d() {
        let mut tensor: Tensor = Tensor::new_from_vec_1d(&[2, 3], &vec![1, 2, 3, 4, 5, 6]);
        tensor.reshape(&[6]);
        assert_eq!(tensor.shape, [6]);
        assert_eq!(tensor.strides, [1]);
        assert_eq!(tensor.dims(), 1);
    }

    #[test]
    fn test_index() {
        let t: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 4]);
        assert_eq!(t[0], 1.0);
        assert_eq!(t[1], 2.0);
        assert_eq!(t[2], 3.0);
        assert_eq!(t[3], 4.0);
    }

    #[test]
    fn test_index_mut() {
        let t: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 4]);
        let mut idx: usize = 0;
        assert_eq!(t[idx], 1.0);
        idx += 1;
        assert_eq!(t[idx], 2.0);
        idx += 1;
        assert_eq!(t[idx], 3.0);
        idx += 1;
        assert_eq!(t[idx], 4.0);
    }

    #[test]
    fn test_at() {
        let t: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 4]);
        assert_eq!(t.at(&[0, 1]).unwrap_err(), 2.0);
        let sub_t: Tensor = t.at(&[0]).unwrap();
        assert_eq!(sub_t.data, vec![1., 2.]);
    }

    #[test]
    fn test_elementwise_add() {
        let a: Tensor = Tensor::new(&[2], &[1, 2]);
        let b: Tensor = Tensor::new(&[2], &[3, 4]);
        let c: Tensor = &a + &b;
        assert_eq!(c.data, &[4.0, 6.0]);
    }

    #[test]
    fn test_elementwise_sub() {
        let a: Tensor = Tensor::new(&[2], &[1, 4]);
        let b: Tensor = Tensor::new(&[2], &[3, 2]);
        let c: Tensor = &a - &b;
        assert_eq!(c.data, &[-2.0, 2.0]);
    }

    #[test]
    fn test_elementwise_mul() {
        let a: Tensor = Tensor::new(&[2], &[1, 2]);
        let b: Tensor = Tensor::new(&[2], &[3, 4]);
        let c: Tensor = &a * &b;
        assert_eq!(c.data, &[3.0, 8.0]);
    }

    #[test]
    fn test_elementwise_div() {
        let a: Tensor = Tensor::new(&[2], &[1, 2]);
        let b: Tensor = Tensor::new(&[2], &[3, 4]);
        let c: Tensor = &a / &b;
        assert_eq!(c.data, &[1.0 / 3.0, 0.5]);
    }

    #[test]
    fn test_elementwise_rem() {
        let a: Tensor = Tensor::new(&[2], &[1, 5]);
        let b: Tensor = Tensor::new(&[2], &[3, 2]);
        let c: Tensor = &a % &b;
        assert_eq!(c.data, &[1.0, 1.0]);
    }
}
