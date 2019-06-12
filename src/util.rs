/// Computes the sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    use std::f64::consts::E;
    1.0 / (1.0 + E.powf(-x))
}

/// Computes the derivative of the sigmoid function
pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn find_index_of_max(a: &[f64]) -> usize {
    use std::f64;
    a.iter()
        .enumerate()
        .fold((std::usize::MAX, f64::NAN), |(i1, x), (i2, y)| {
            if x > *y {
                (i1, x)
            } else {
                (i2, *y)
            }
        })
        .0
}

pub fn index_to_output_layer(i: usize, output_size: usize) -> Vec<f64> {
    let mut vec = vec![0.0; output_size];
    vec[i] = 1.0;

    vec
}
