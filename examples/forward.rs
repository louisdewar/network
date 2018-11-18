extern crate network;
use network::Network;

pub fn main() {
    let layer_sizes = vec![2, 3, 2];
    let weights = vec![0.5, 0.2, 0.1, 0.3, 0.7, 0.6, 0.4, 0.9, 0.4, 0.9, 0.1, 0.2];
    let biases = vec![0.7, 0.2, 0.1, 0.9, 0.8];

    let network = Network::new(layer_sizes, weights, biases);

    let output = network.feed_forward(vec![5.0, 1.0]);
    let entire = network.feed_forward_entire(vec![5.0, 1.0]);

    // println!("Output: {:?}", output);
    //
    // println!("Entire output: {:?}", entire);

    // Check that the two output neurons equal, the last layer will be the last entries in the entire output
    assert!(
        output[0] == entire[entire.len() - 2] && output[1] == entire[entire.len() - 1],
        "Outputs didn't equal each other"
    );
}
