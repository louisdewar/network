/// Computes the sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    use std::f64::consts::E;
    1.0 / (1.0 + E.powf(-x))
}

/// Computes the derivative of the sigmoid function
pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
