extern crate network;

use network::Network;

pub fn main() {
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

        println!(
            "Error: {}, Got output: {:?}, expected output: {:?}",
            output[0] - desired_output[0],
            output,
            desired_output
        )
    }
    println!("{}", network.examine_error(training_examples.clone()));

    let average_error = network.train(training_examples.clone(), 100000, 0.01);

    println!("Average error: {}", average_error);

    println!("===After training===");
    for (input, desired_output) in training_examples.clone() {
        let output = network.feed_forward(input);

        let correct =
            network::find_index_of_max(&output) == network::find_index_of_max(&desired_output);

        println!(
            "{}    Error: {}, Got output: {:?}, expected output: {:?}",
            if correct { "✅" } else { "❌" },
            output[0] - desired_output[0],
            network::find_index_of_max(&output),
            network::find_index_of_max(&desired_output),
        );
    }

    println!("{}", network.examine_error(training_examples));
}
