extern crate rand;

mod util;
pub use util::find_index_of_max;

pub struct Network {
    // Where the weights for the [i + 1]th (we skip the input layer) layer begin in the array
    weights_indices: Vec<usize>,
    weights: Vec<f64>,
    // Where the biases for the [i + 1]th layer begin
    biases_indices: Vec<usize>,
    biases: Vec<f64>,
    layer_sizes: Vec<usize>,
    // Cumulative frequency of neurons per layer
    neuron_indices: Vec<usize>,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>, weights: Vec<f64>, biases: Vec<f64>) -> Network {
        let mut weights_indices = Vec::new();
        let mut biases_indices = Vec::new();
        let mut neuron_indices = vec![0];

        let mut weight_acc = 0;
        let mut bias_acc = 0;
        let mut neuron_acc = layer_sizes[0];

        // Ignore the first layer, as it is the input and has no weight
        for i in 1..layer_sizes.len() {
            weights_indices.push(weight_acc);
            biases_indices.push(bias_acc);
            neuron_indices.push(neuron_acc);

            // The number of weights of a neuron in a layer is the number of neurons in the previous layer
            // And so the total number of weights in the entire layer is the number of weights per neuron * the number of neurons
            weight_acc += layer_sizes[i - 1] * layer_sizes[i];

            // The number of biases in a layer is simply the number of neurons in the layer
            bias_acc += layer_sizes[i];

            // Sum the neurons (similar to bias but it includes the first layer)
            neuron_acc += layer_sizes[i];
        }

        assert!(
            weights.len() == weight_acc,
            "Incorrect number of weights given in network creation"
        );
        assert!(
            biases.len() == bias_acc,
            "Incorrect number of biases given in network creation"
        );

        Network {
            weights,
            biases,
            layer_sizes,
            weights_indices,
            biases_indices,
            neuron_indices,
        }
    }

    pub fn generate_random(layer_sizes: Vec<usize>) -> Network {
        use rand::distributions::Normal;
        use rand::prelude::*;

        let mut weights = Vec::new();
        let mut n_biases = 0;

        for i in 1..layer_sizes.len() {
            let mut new_weights = rand::thread_rng()
                // Weights are distributed as normal with mean 0 and variance of 1/(number of weights connecting to a single neuron)
                .sample_iter(&Normal::new(0.0, 1.0 / layer_sizes[i - 1] as f64))
                .take(layer_sizes[i] * layer_sizes[i - 1])
                .collect::<Vec<f64>>();

            weights.append(&mut new_weights);

            n_biases += layer_sizes[i];
        }

        let biases = rand::thread_rng()
            .sample_iter(&Normal::new(0.0, 1.0))
            .take(n_biases)
            .collect();

        Network::new(layer_sizes, weights, biases)
    }

    pub fn feed_forward(&self, inputs: Vec<f64>) -> Vec<f64> {
        assert!(
            inputs.len() == self.layer_sizes[0],
            "Incorrect number of inputs given in feed forward"
        );

        // TODO: Investigate re-using two vectors to store data, with capacity for largest layer
        let mut activations = inputs;

        for layer_index in 1..self.layer_sizes.len() {
            let weight_index = self.weights_indices[layer_index - 1];
            let bias_index = self.biases_indices[layer_index - 1];
            let layer_size = self.layer_sizes[layer_index];

            let mut zs = Vec::with_capacity(layer_size);

            // Tells us how many weights per neuron
            let prev_layer_size = self.layer_sizes[layer_index - 1];

            for neuron_index in 0..layer_size {
                // Get the start of the weights for this specific neuron
                // START_OF_WEIGHTS_FOR_THIS_LAYER + OFFSET_FOR_SPECIFIC_NEURON
                let weight_start_index = weight_index + neuron_index * prev_layer_size;

                zs.push(
                    (0..prev_layer_size)
                        .map(|i| self.weights[weight_start_index + i] * activations[i])
                        .sum(),
                );
            }

            activations = zs
                .into_iter()
                .enumerate()
                .map(|(i, z): (usize, f64)| util::sigmoid(z + self.biases[bias_index + i]))
                .collect();
        }

        activations
    }

    pub fn feed_forward_entire(&self, inputs: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        assert!(
            inputs.len() == self.layer_sizes[0],
            "Incorrect number of inputs given in feed forward"
        );

        let mut activations = Vec::with_capacity(self.biases.len());
        let mut zs = Vec::with_capacity(self.biases.len());

        // The activations of the previous layer
        let mut last_layer = inputs;

        for layer_index in 1..self.layer_sizes.len() {
            let weight_index = self.weights_indices[layer_index - 1];
            let bias_index = self.biases_indices[layer_index - 1];
            let layer_size = self.layer_sizes[layer_index];

            // Tells us how many weights per neuron
            let prev_layer_size = self.layer_sizes[layer_index - 1];

            for neuron_index in 0..layer_size {
                // Get the start of the weights for this specific neuron
                // START_OF_WEIGHTS_FOR_THIS_LAYER + OFFSET_FOR_SPECIFIC_NEURON
                let weight_start_index = weight_index + neuron_index * prev_layer_size;

                zs.push(
                    (0..prev_layer_size)
                        .map(|i| self.weights[weight_start_index + i] * last_layer[i])
                        .sum(),
                );
            }

            activations.append(&mut last_layer);

            last_layer = zs[bias_index..bias_index + layer_size]
                .iter()
                .enumerate()
                .map(|(i, z): (usize, &f64)| util::sigmoid(z + self.biases[bias_index + i]))
                .collect();
        }

        // Add the output layer
        activations.append(&mut last_layer);

        (activations, zs)
    }

    /// Returns activations and the vector of deltas (the amount that each weight needs to be adjusted)
    pub fn back_propogate(&self, input: Vec<f64>, desired_output: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut deltas = vec![0.0; self.biases.len()];
        // All the activations, excluding the input layer
        let (activations, zs) = self.feed_forward_entire(input);

        // Calculate the error for the output layer
        let mut last_error = {
            let layer_index = self.layer_sizes.len() - 1;
            let activation_start = self.neuron_indices[layer_index];
            let delta_start = self.biases_indices[layer_index - 1];
            let layer_size = self.layer_sizes[layer_index];

            // Cost function for output layer
            let error = (activation_start..activation_start + layer_size)
                .zip(desired_output.iter())
                .map(|(i, y)| {
                    let a = activations[i];
                    // This is the definition of the cost function (currently cross-entropy)
                    (a - y)
                    // Another cost function
                    // (activations[i] - y) * util::sigmoid_prime(zs[i - self.layer_sizes[0]])
                });

            for (delta, error) in deltas[delta_start..delta_start + layer_size]
                .iter_mut()
                .zip(error)
            {
                *delta = error
            }

            (delta_start..delta_start + layer_size)
        };

        // Reverse through the hidden layers (0 becomes first layer)
        for layer_index in (0..self.layer_sizes.len() - 1).rev() {
            let layer_size = self.layer_sizes[layer_index + 1];
            // prev layer here means the last layer from the loop (even though it's actually the next layer)
            let prev_layer_weight_start = self.weights_indices[layer_index];

            // Error for the previous layer
            let error_start = self.biases_indices[layer_index];

            for (prev_neuron_i, prev_err_i) in last_error.enumerate() {
                let start = prev_layer_weight_start + prev_neuron_i * layer_size;
                for (cur_neuron_i, new_err_i) in (error_start..error_start + layer_size).enumerate()
                {
                    deltas[new_err_i] += self.weights[start + cur_neuron_i]
                        * deltas[prev_err_i]
                        * util::sigmoid_prime(
                            zs[error_start + prev_neuron_i],
                        );
                }
            }

            last_error = error_start..error_start + layer_size;
        }

        (activations, deltas)
    }

    fn mini_batch(&mut self, mini_batch: Vec<(Vec<f64>, Vec<f64>)>, n: usize, learn_rate: f64, lambda: f64) {
        // Learn rate should be averaged across batch
        let learn_rate = learn_rate / mini_batch.len() as f64;

        let mut total_activations =
            vec![0.0; self.neuron_indices.last().unwrap() + self.layer_sizes.last().unwrap()];
        let mut total_deltas = vec![0.0; self.biases.len()];

        let mini_batch_len = mini_batch.len() as f64;

        for (input, desired_output) in mini_batch {
            let (activations, deltas) = self.back_propogate(input.to_vec(), &desired_output);

            for (total, activation) in total_activations.iter_mut().zip(activations.iter()) {
                *total += activation;
            }

            for (total, delta) in total_deltas.iter_mut().zip(deltas.iter()) {
                *total += delta;
            }
        }

        for layer_index in 1..self.layer_sizes.len() {
            let input = {
                let start = self.neuron_indices[layer_index - 1];
                &total_activations[start..start + self.layer_sizes[layer_index - 1]]
            };

            let delta_start = self.biases_indices[layer_index - 1];
            for (neuron_i, delta) in total_deltas
                [delta_start..delta_start + self.layer_sizes[layer_index]]
                .iter()
                .enumerate()
            {
                let weight_start = self.weights_indices[layer_index - 1]
                    + self.layer_sizes[layer_index - 1] * neuron_i;
                for (weight_i, weight) in self.weights
                    [weight_start..weight_start + self.layer_sizes[layer_index - 1]]
                    .iter_mut()
                    .enumerate()
                {
                    *weight = *weight * (1.0 - learn_rate * (lambda / n as f64)) - (learn_rate / mini_batch_len) * input[weight_i] * delta * learn_rate;
                }

                // Update bias
                self.biases[self.biases_indices[layer_index - 1] + neuron_i] -= (learn_rate / mini_batch_len) * delta * learn_rate;
            }
        }
    }

    pub fn examine_error(&self, test_data: Vec<(Vec<f64>, Vec<f64>)>) -> f64 {
        let mut sq_error = 0.0;

        let test_data_len = test_data.len();

        for (input, desired_output) in test_data {
            let output = self.feed_forward(input);

            let average_sq_error = output
                .into_iter()
                .zip(desired_output.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                / desired_output.len() as f64;
            sq_error += average_sq_error;
        }

        sq_error / test_data_len as f64
    }

    pub fn batch_train(
        &mut self,
        training_examples: Vec<(Vec<f64>, Vec<f64>)>,
        epochs: usize,
        batch_size: usize,
        learn_rate: f64,
        test_data: Option<&Vec<(Vec<f64>, Vec<f64>)>>,
    ) {
        for epoch in 0..epochs {
            for batch in training_examples.chunks(batch_size) {
                // TOOD: pick suitable lambda
                self.mini_batch(batch.to_vec(), training_examples.len(), learn_rate, 5.0);
            }

            if epoch % 5 == 0 {
                if let Some(data) = &test_data {
                    println!("Epoch {}, err: {}", epoch, self.examine_error(data.to_vec()));
                } else {
                    println!("Epoch {}", epoch);
                }
            }
        }
    }

    pub fn train(
        &mut self,
        training_examples: Vec<(Vec<f64>, Vec<f64>)>,
        epochs: usize,
        learn_rate: f64,
    ) -> f64 {
        let mut error = 0.0;

        for epoch in 0..epochs {
            for (input, desired_output) in &training_examples {
                let (activations, deltas) = self.back_propogate(input.to_vec(), desired_output);

                // Loop through all layers except for start
                for layer_index in 1..self.layer_sizes.len() {
                    let input = {
                        let start = self.neuron_indices[layer_index - 1];
                        &activations[start..start + self.layer_sizes[layer_index - 1]]
                    };

                    let delta_start = self.biases_indices[layer_index - 1];
                    for (neuron_i, delta) in deltas
                        [delta_start..delta_start + self.layer_sizes[layer_index]]
                        .iter()
                        .enumerate()
                    {
                        let weight_start = self.weights_indices[layer_index - 1]
                            + self.layer_sizes[layer_index - 1] * neuron_i;
                        for (weight_i, weight) in self.weights
                            [weight_start..weight_start + self.layer_sizes[layer_index - 1]]
                            .iter_mut()
                            .enumerate()
                        {
                            *weight -= input[weight_i] * delta * learn_rate;
                        }

                        // Update bias
                        self.biases[self.biases_indices[layer_index - 1] + neuron_i] -=
                            delta * learn_rate;
                    }
                }

                // If last epoch
                if epoch == epochs - 1 {
                    error = {
                        let output_start = *self.neuron_indices.last().unwrap();
                        activations[output_start..output_start + *self.layer_sizes.last().unwrap()]
                            .iter()
                            .zip(desired_output.iter())
                            .map(|(output, desired)| (output - desired).powi(2))
                            .sum::<f64>()
                    };
                }
            }
        }

        error / training_examples.len() as f64
    }
}
