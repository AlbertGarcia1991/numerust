#![allow(dead_code)] // Suppress "is never used" warnings globally in this file
#![allow(non_snake_case)]

use std::ops::{Add, BitXor, Div, Index, IndexMut, Mul, Rem, Sub};

#[derive(Debug)]
enum ReductionType {
    Max,
    Min,
    Med,
    Sum,
    Avg,
    Std,
}

enum ReductionAxis {
    Absolute,
    Axis(Vec<usize>),
}

#[derive(Clone, Debug)]
struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides: Vec<usize> = vec![1; shape.len()];
    if shape.is_empty() {
        return Vec::new();
    } else {
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
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

    pub fn new_from_vec_1d<T>(shape: &[usize], data: &[T]) -> Self
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

    pub fn at(&self, indices: &[usize]) -> Tensor {
        let mut flat_index: usize = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                panic!("Index out of bounds for dimension {}", i);
            }
            flat_index += idx * self.strides[i];
        }
        // If all dimensions are specified, return the single value
        if indices.len() == self.shape.len() {
            return Self::new_1d(&[self.data[flat_index]]);
        }
        // Otherwise, calculate the sub-tensor
        let sub_shape: Vec<usize> = self.shape[indices.len()..].to_vec();
        let sub_strides: Vec<usize> = self.strides[indices.len()..].to_vec();
        let sub_size: usize = prod(&sub_shape);
        let sub_data: Vec<f64> = self.data[flat_index..flat_index + sub_size].to_vec();
        Tensor {
            data: sub_data,
            shape: sub_shape,
            strides: sub_strides,
        }
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

    pub fn reduce(&self, reduction_type: ReductionType, reduction_axis: ReductionAxis) -> Self {
        // ---------- 1. Normalise and validate axis argument ----------
        let mut axes: Vec<usize> = match reduction_axis {
            ReductionAxis::Absolute => (0..self.dims()).collect(),
            ReductionAxis::Axis(ax) if !ax.is_empty() => {
                let mut ax: Vec<usize> = ax.clone();
                ax.sort_unstable();
                ax.dedup();
                if ax.iter().any(|&a| a >= self.dims()) {
                    panic!("Axis out of bounds");
                }
                ax
            }
            _ => panic!("Axis list cannot be empty"),
        };

        // Fast path: reduce over every axis ⇒ scalar result
        if axes.len() == self.dims() {
            return Tensor::new_1d(&[self.reduction_op(&reduction_type)]);
        }

        // ---------- 2. Recursive helper that uses `at` ----------
        fn reduce_recursive(
            tensor: &Tensor,
            axes: Vec<usize>,
            reduction_type: &ReductionType,
        ) -> Tensor {
            if axes.is_empty() {
                return tensor.clone(); // nothing left to reduce
            }

            // Always look at the first axis in the (sorted) list
            let first_axis: usize = axes[0];

            /* ---- Case A : we are reducing the leading axis (0) ---- */
            if first_axis == 0 {
                // Collect every slice [i] and perform an element-wise reduction
                let slices: Vec<Tensor> = (0..tensor.shape[0]).map(|i| tensor.at(&[i])).collect();

                let elem_cnt: usize = slices[0].data.len();
                let mut out_data: Vec<f64> = Vec::with_capacity(elem_cnt);

                for elem in 0..elem_cnt {
                    let mut bucket: Vec<f64> = Vec::with_capacity(slices.len());
                    for s in &slices {
                        bucket.push(s.data[elem]);
                    }
                    out_data.push(Tensor::new_1d(&bucket).reduction_op(reduction_type));
                }

                // Result shape = slice-shape with axis-0 removed
                let mut out_shape: Vec<usize> = slices[0].shape.clone();
                if out_shape.len() == 1 && out_shape[0] == 1 {
                    out_shape.clear(); // collapse to scalar when needed
                }
                let reduced: Tensor = Tensor::new(&out_shape, &out_data);

                // Still more axes to reduce?  Shift them down and recurse.
                let next_axes: Vec<usize> = axes[1..].iter().map(|a| a - 1).collect();
                if next_axes.is_empty() {
                    reduced
                } else {
                    reduce_recursive(&reduced, next_axes, reduction_type)
                }
            }
            /* ---- Case B : leading axis is *not* reduced ------------ */
            else {
                // Every slice stays at the front; we recurse inside each one
                let new_axes: Vec<usize> = axes.iter().map(|a| a - 1).collect(); // shift for sub-tensor

                let mut reduced_slices: Vec<Tensor> = Vec::with_capacity(tensor.shape[0]);
                for i in 0..tensor.shape[0] {
                    let sub: Tensor = tensor.at(&[i]);
                    reduced_slices.push(reduce_recursive(&sub, new_axes.clone(), reduction_type));
                }

                // Stack the reduced slices back together
                let slice_shape = &reduced_slices[0].shape;
                let mut out_shape = vec![reduced_slices.len()];
                if !slice_shape.is_empty() {
                    out_shape.extend_from_slice(slice_shape);
                }
                let mut out_data = Vec::with_capacity(prod(&out_shape));
                for s in reduced_slices {
                    out_data.extend_from_slice(&s.data);
                }
                Tensor::new(&out_shape, &out_data)
            }
        }

        axes.sort_unstable();
        reduce_recursive(self, axes, &reduction_type)
    }

    fn reduction_op(&self, reduction_type: &ReductionType) -> f64 {
        match reduction_type {
            ReductionType::Max => self
                .data
                .iter()
                .copied()
                .max_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap())
                .unwrap(),
            ReductionType::Min => self
                .data
                .iter()
                .copied()
                .min_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap())
                .unwrap(),
            ReductionType::Sum => self.data.iter().copied().sum(),
            ReductionType::Med => {
                let mut sorted: Vec<f64> = self
                    .data
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan()) // Optional: ignore NaNs
                    .collect();
                let len: usize = sorted.len();
                sorted.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap()); // NaN-safe sort

                let mid: usize = len / 2;
                if len % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
            ReductionType::Avg => self.reduction_op(&ReductionType::Sum) / (self.data.len() as f64),
            ReductionType::Std => {
                let avg: f64 = self.reduction_op(&ReductionType::Avg);
                ((&(self - avg) ^ 2).reduction_op(&ReductionType::Sum)
                    / ((self.data.len() - 1) as f64))
                    .sqrt()
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
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
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
        let tensor: Tensor = Tensor::new_from_vec_1d(&[2, 3], &[1, 2, 3, 4]);
        assert_eq!(tensor.data, vec![1., 2., 3., 4.]);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.strides, vec![2, 1]);
        assert_eq!(tensor.dims(), 2);
    }

    #[test]
    fn test_reshape() {
        let mut tensor: Tensor = Tensor::new_from_vec_1d(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        tensor.reshape(&[3, 2]);
        assert_eq!(tensor.shape, [3, 2]);
        assert_eq!(tensor.strides, [2, 1]);
    }

    #[test]
    fn test_reshape_from_1d() {
        let mut tensor: Tensor = Tensor::new_from_vec_1d(&[1, 6], &[1, 2, 3, 4, 5, 6]);
        tensor.reshape(&[3, 2]);
        assert_eq!(tensor.shape, [3, 2]);
        assert_eq!(tensor.strides, [2, 1]);
    }

    #[test]
    fn test_reshape_to_1d() {
        let mut tensor: Tensor = Tensor::new_from_vec_1d(&[2, 3], &[1, 2, 3, 4, 5, 6]);
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
    fn test_at_2d() {
        let t: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 4]);
        assert_eq!(t.at(&[0, 1]).data, [2.0]);
        let sub_t: Tensor = t.at(&[0]);
        assert_eq!(sub_t.data, [1., 2.]);
        let sub_t: Tensor = t.at(&[1]);
        assert_eq!(sub_t.data, [3., 4.]);
    }

    #[test]
    fn test_at_3d() {
        let t: Tensor = Tensor::new(&[2, 2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(t.at(&[0]), Tensor::new(&[2, 2], &[1, 2, 3, 4]));
        assert_eq!(t.at(&[1]), Tensor::new(&[2, 2], &[5, 6, 7, 8]));
        assert_eq!(t.at(&[0, 0]), Tensor::new(&[2], &[1, 2]));
        assert_eq!(t.at(&[0, 1]), Tensor::new(&[2], &[3, 4]));
        assert_eq!(t.at(&[1, 0]), Tensor::new(&[2], &[5, 6]));
        assert_eq!(t.at(&[1, 1]), Tensor::new(&[2], &[7, 8]));
        assert_eq!(t.at(&[0, 0, 0]).data, [1.0]);
        assert_eq!(t.at(&[0, 1, 0]).data, [3.0]);
        assert_eq!(t.at(&[1, 0, 0]).data, [5.0]);
    }

    #[test]
    fn test_eq() {
        let a: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 4]);
        let b: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 4]);
        assert!(a == b);
        let c: Tensor = Tensor::new(&[2, 2], &[1, 2, 3, 5]);
        assert!(a != c);
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

    #[test]
    fn test_max() {
        let data: Tensor = Tensor::new_1d(&[1.0, 5.0, -3.0]);
        assert_eq!(data.reduction_op(&ReductionType::Max), 5.0);
    }

    #[test]
    fn test_min() {
        let data: Tensor = Tensor::new_1d(&[1.0, 5.0, -3.0]);
        assert_eq!(data.reduction_op(&ReductionType::Min), -3.0);
    }

    #[test]
    fn test_sum() {
        let data: Tensor = Tensor::new_1d(&[1.0, 5.0, -3.0]);
        assert_eq!(data.reduction_op(&ReductionType::Sum), 3.0);
    }

    #[test]
    fn test_med_odd() {
        let data: Tensor = Tensor::new_1d(&[1.0, 5.0, -3.0]);
        assert_eq!(data.reduction_op(&ReductionType::Med), 1.0);
    }

    #[test]
    fn test_med_even() {
        let data: Tensor = Tensor::new_1d(&[1.0, 5.0, -3.0, 0.0]);
        assert_eq!(data.reduction_op(&ReductionType::Med), 0.5);
    }

    #[test]
    fn test_avg() {
        let data: Tensor = Tensor::new_1d(&[1.0, 5.0, -3.0]);
        assert_eq!(data.reduction_op(&ReductionType::Avg), 1.0);
    }

    #[test]
    fn test_std() {
        let data: Tensor = Tensor::new_1d(&[1.0, 5.0, -3.0]);
        let std: f64 = data.reduction_op(&ReductionType::Std);
        assert_eq!(std, 4.0);
    }

    // ---------- helpers ----------
    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn scalar(t: Tensor) -> f64 {
        assert_eq!(t.shape, vec![1]);
        t.data[0]
    }

    #[test]
    fn reduce_1d_all_types() {
        let t: Tensor = Tensor::new_1d(&[1.0, 3.0, -2.0, 4.0]); // len = 4
        assert_eq!(
            scalar(t.reduce(ReductionType::Max, ReductionAxis::Absolute)),
            4.0
        );
        assert_eq!(
            scalar(t.reduce(ReductionType::Min, ReductionAxis::Absolute)),
            -2.0
        );
        assert_eq!(
            scalar(t.reduce(ReductionType::Sum, ReductionAxis::Absolute)),
            6.0
        );
        assert!(approx_eq(
            scalar(t.reduce(ReductionType::Avg, ReductionAxis::Absolute)),
            1.5,
            1e-12
        ));
        assert_eq!(
            scalar(t.reduce(ReductionType::Med, ReductionAxis::Absolute)),
            2.0
        );
        assert!(approx_eq(
            scalar(t.reduce(ReductionType::Std, ReductionAxis::Absolute)),
            7f64.sqrt(),
            1e-12
        ));
        assert_eq!(
            scalar(t.reduce(ReductionType::Max, ReductionAxis::Axis(vec![0]))),
            4.0
        );
        assert!(approx_eq(
            scalar(t.reduce(ReductionType::Std, ReductionAxis::Axis(vec![0]))),
            7f64.sqrt(),
            1e-12
        ));
    }

    #[test]
    fn reduce_2d_axis0() {
        let t: Tensor = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r: Tensor = t.reduce(ReductionType::Max, ReductionAxis::Axis(vec![0]));
        assert_eq!(r, Tensor::new_1d(&[4.0, 5.0, 6.0]));
        let r: Tensor = t.reduce(ReductionType::Sum, ReductionAxis::Axis(vec![0]));
        assert_eq!(r, Tensor::new_1d(&[5.0, 7.0, 9.0]));
    }

    #[test]
    fn reduce_2d_axis1() {
        let t: Tensor = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r: Tensor = t.reduce(ReductionType::Min, ReductionAxis::Axis(vec![1]));
        assert_eq!(r, Tensor::new_1d(&[1.0, 4.0]));
        let r: Tensor = t.reduce(ReductionType::Avg, ReductionAxis::Axis(vec![1]));
        assert_eq!(r.shape, vec![2]);
        assert!(approx_eq(r.data[0], 2.0, 1e-12));
        assert!(approx_eq(r.data[1], 5.0, 1e-12));
    }

    #[test]
    fn reduce_3d_multiple_axes() {
        let t: Tensor = Tensor::new(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let r: Tensor = t.reduce(ReductionType::Max, ReductionAxis::Axis(vec![0, 2]));
        assert_eq!(r, Tensor::new_1d(&[6.0, 8.0]));
        let r: Tensor = t.reduce(ReductionType::Std, ReductionAxis::Axis(vec![1]));
        let expected: Tensor = Tensor::new(&[2, 2], &[2_f64.sqrt(); 4]);
        assert!(r
            .data
            .iter()
            .zip(&expected.data)
            .all(|(&a, &b)| approx_eq(a, b, 1e-9)));
    }
}
