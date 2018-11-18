#[macro_use]
extern crate criterion;

extern crate network;
use network::Network;

use criterion::Criterion;

fn small() {
    let layer_sizes = vec!(2, 3, 2);
    let weights = vec!(0.5, 0.2, 0.1, 0.3, 0.7, 0.6, 0.4, 0.9, 0.4, 0.9, 0.1, 0.2);
    let biases = vec!(0.7, 0.2, 0.1, 0.9, 0.8);

    let network = Network::new(layer_sizes, weights, biases);

    let output = network.feed_forward(vec!(5.0, 1.0));

    assert!(output == [0.9120659277720782, 0.8744713083223637]);

}

fn small_entire() {
    let layer_sizes = vec!(2, 3, 2);
    let weights = vec!(0.5, 0.2, 0.1, 0.3, 0.7, 0.6, 0.4, 0.9, 0.4, 0.9, 0.1, 0.2);
    let biases = vec!(0.7, 0.2, 0.1, 0.9, 0.8);

    let network = Network::new(layer_sizes, weights, biases);

    let output = network.feed_forward_entire(vec!(5.0, 1.0));

    // assert!(output == ([0.9677045353015495, 0.7310585786300049, 0.9852259683067269, 0.9120659277720782, 0.8744713083223637]));
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("forward small entire", |b| b.iter(|| small_entire()));
    c.bench_function("forward small", |b| b.iter(|| small()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
