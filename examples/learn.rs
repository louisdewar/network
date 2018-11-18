extern crate network;
// extern crate rand;
//
// use rand::prelude::*;
// use rand::distributions::Standard;

use network::Network;

// fn compare_expected_to_output(expected: Vec<f64>, output: Vec<f64>) {
//
// }

pub fn main() {
    // let layer_sizes = vec![2, 3, 2, 1];
    // let weights = vec![0.5, 0.2, 0.1, 0.3, 0.7, 0.6, 0.4, 0.9, 0.4, 0.4, 0.9, 0.4, 0.4, 0.9];
    // let biases = vec![0.7, 0.2, 0.1, 0.9, 0.2, 0.5];
    //
    // let mut network = Network::new(layer_sizes, weights, biases);

    // let mut least_error = std::f64::MAX;

    // for i in 0..1000 {
        let layer_sizes = vec![2, 10, 2];

        let mut network = Network::generate_random(layer_sizes);

        let training_examples = vec![
            // First neuron in output is '0' second is '1'
            (vec![0.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0], vec![0.0, 1.0]),
            (vec![1.0, 0.0], vec![0.0, 1.0]),
            (vec![1.0, 1.0], vec![1.0, 0.0]),
        ];

        println!("===Before training===");
        for (input, desired_output) in training_examples.clone() {
            let output = network.feed_forward(input);

            println!("Error: {}, Got output: {:?}, expected output: {:?}", output[0] - desired_output[0], output, desired_output)
        }
        println!("{}", network.examine_error(training_examples.clone()));

        let average_error = network.train(training_examples.clone(), 10000, 0.013);

        // if average_error < least_error {
        //     least_error = average_error;
        // }

        println!("Average error: {}", average_error);


        println!("===After training===");
        for (input, desired_output) in training_examples.clone() {
            let output = network.feed_forward(input);

            println!("Error: {}, Got output: {:?}, expected output: {:?}", output[0] - desired_output[0], output, desired_output);
        }

        println!("{}", network.examine_error(training_examples));
    // }

    // println!("Least error: {}", least_error);
}
