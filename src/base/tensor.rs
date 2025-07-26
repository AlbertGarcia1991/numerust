#![allow(dead_code)] // Suppress "is never used" warnings globally in this file
#![allow(non_snake_case)]

use std::ops::{Add, BitXor, Div, Index, IndexMut, Mul, Rem, Sub};

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

    pub fn new_1d<T>(values: &[T]) -> Self
    where
        T: Into<f64> + Copy,
    {
        let shape: [usize; 1] = [values.len()];
        let mut data: Vec<f64> = Vec::with_capacity(values.len());
        for &v in values {
            data.push(v.into());
        }
        let strides: Vec<usize> = compute_strides(&shape);
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

    pub fn T(&self) -> Tensor {
        if self.dims() == 1 {
            panic!("Transpose requires dims > 1");
        }

        Tensor {
            data: self.data.clone(),
            shape: self.shape.iter().rev().cloned().collect(),
            strides: self.strides.iter().rev().cloned().collect(),
        }
    }

    // TODO: Make agnostic to number of dimensions
    pub fn max(&self, axis: Option<Tensor>) -> Result<Tensor, f64> {
        match axis {
            None => {
                // Global max (no axis specified)
                let max_val: f64 = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                Err(max_val)
            }
            Some(axis_tensor) => {
                // Max along specified axis
                if axis_tensor.shape.len() != 1 {
                    panic!("Axis tensor must be 1-dimensional");
                }

                let axis: usize = axis_tensor.data[0] as usize;
                if axis >= self.shape.len() {
                    panic!("Axis out of bounds");
                }

                // Calculate the shape of the result tensor
                let mut new_shape: Vec<usize> = self.shape.clone();
                new_shape.remove(axis);

                let mut max_values = Vec::new();

                // For a 2D tensor with shape [M, N]:
                // If axis = 0, we want max of each column (max across rows)
                // If axis = 1, we want max of each row (max across columns)
                if axis == 0 {
                    // Process each row (find max across columns)
                    for row in 0..self.shape[0] {
                        let mut max_val = f64::NEG_INFINITY;
                        // Look at each column in this row
                        for col in 0..self.shape[1] {
                            let idx = row * self.strides[0] + col;
                            max_val = max_val.max(self.data[idx]);
                        }
                        max_values.push(max_val);
                    }
                } else {
                    // Process each column (find max across rows)
                    for col in 0..self.shape[1] {
                        let mut max_val = f64::NEG_INFINITY;
                        // Look at each row in this column
                        for row in 0..self.shape[0] {
                            let idx = row * self.strides[0] + col;
                            max_val = max_val.max(self.data[idx]);
                        }
                        max_values.push(max_val);
                    }
                }

                Ok(Tensor::new(&new_shape, &max_values))
            }
        }
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
        impl<T> $trait<T> for &Tensor
        where
            T: Into<f64> + Copy,
        {
            type Output = Tensor;
            fn $func(self, rhs: T) -> Self::Output {
                let rhs_f64 = rhs.into();
                let data = self.data.iter().map(|a| a.$func(rhs_f64)).collect();
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

impl BitXor for &Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.powf(*b))
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> BitXor<T> for &Tensor
where
    T: Into<f64> + Copy,
{
    type Output = Tensor;
    fn bitxor(self, rhs: T) -> Self::Output {
        let rhs_f64: f64 = rhs.into();
        let data: Vec<f64> = self.data.iter().map(|a| a.powf(rhs_f64)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
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

    #[test]
    fn test_elementwise_add_scalar() {
        let a: Tensor = Tensor::new(&[2], &[1, 2]);
        let b: Tensor = &a + 10.;
        assert_eq!(b.data, &[11.0, 12.0]);
        let c: Tensor = &a + 10;
        assert_eq!(c.data, &[11.0, 12.0]);
    }

    #[test]
    fn test_elementwise_sub_scalar() {
        let a: Tensor = Tensor::new(&[2], &[1, 2]);
        let b: Tensor = &a - 1;
        assert_eq!(b.data, &[0.0, 1.0]);
    }

    #[test]
    fn test_elementwise_mul_scalar() {
        let a: Tensor = Tensor::new(&[2], &[1, 2]);
        let b: Tensor = &a * 3;
        assert_eq!(b.data, &[3.0, 6.0]);
    }

    #[test]
    fn test_elementwise_div_scalar() {
        let a: Tensor = Tensor::new(&[2], &[1, 3]);
        let b: Tensor = &a / 2;
        assert_eq!(b.data, &[0.5, 1.5]);
    }

    #[test]
    fn test_elementwise_mod_scalar() {
        let a: Tensor = Tensor::new(&[2], &[1, 2]);
        let b: Tensor = &a % 2;
        assert_eq!(b.data, &[1.0, 0.0]);
    }

    #[test]
    fn test_elementwise_exp() {
        let a: Tensor = Tensor::new_1d(&[-1, 0, 2]);
        let b: Tensor = Tensor::new_1d(&[2, 2, 2]);
        let c: Tensor = &a ^ &b;
        assert_eq!(c.data, &[1.0, 0.0, 4.0]);
    }

    #[test]
    fn test_elementwise_exp_scalar() {
        let a: Tensor = Tensor::new(&[2], &[3, 4]);
        let b: Tensor = &a ^ 2;
        assert_eq!(b.data, &[9.0, 16.0]);
    }

    #[test]
    fn test_max_global() {
        let t: Tensor = Tensor::new(&[2, 2], &[1, 5, 3, 4]);
        let m: f64 = t.max(None).unwrap_err();
        assert_eq!(m, 5.0);
    }

    #[test]
    fn test_max_axis() {
        let t: Tensor = Tensor::new(&[2, 2], &[1, 5, 3, 4]);
        let axis_tensor: Tensor = Tensor::new_1d(&[0]); // Specify axis 0
        let m: Tensor = t.max(Some(axis_tensor)).unwrap();
        assert_eq!(m.data, &[5.0, 4.0]); // Max along axis 0
    }

    #[test]
    fn test_max_axis_transposed() {
        let t: Tensor = Tensor::new(&[2, 2], &[1, 5, 3, 4]);
        let axis_tensor: Tensor = Tensor::new_1d(&[1]); // Specify axis 0
        let m: Tensor = t.max(Some(axis_tensor)).unwrap();
        assert_eq!(m.data, &[3.0, 5.0]); // Max along axis 0
    }

    #[test]
    fn test_transpose() {
        let t: Tensor = Tensor::new(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let t_transposed: Tensor = t.T();
        assert_eq!(t_transposed.shape, vec![3, 2]);
        assert_eq!(t_transposed.strides, vec![1, 3]);
        // The data is not rearranged, only shape and strides are changed
        assert_eq!(t_transposed.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_transpose_3d() {
        let t: Tensor = Tensor::new(
            &[2, 3, 4],
            &[
                1, 2, 3, 4, // [0, 0, :]
                5, 6, 7, 8, // [0, 1, :]
                9, 10, 11, 12, // [0, 2, :]
                13, 14, 15, 16, // [1, 0, :]
                17, 18, 19, 20, // [1, 1, :]
                21, 22, 23, 24, // [1, 2, :]
            ],
        );
        let t_transposed: Tensor = t.T();
        assert_eq!(t_transposed.shape, vec![4, 3, 2]);
        assert_eq!(t_transposed.strides, vec![1, 4, 12]);
        // The data is not rearranged, only shape and strides are changed
        assert_eq!(
            t_transposed.data,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
            ]
        );
    }
}
