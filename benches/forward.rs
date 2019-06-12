#[macro_use]
extern crate criterion;

extern crate network;
use network::Network;

use criterion::{black_box, Criterion};

fn small() {
    let layer_sizes = vec![2, 3, 2];
    let weights = vec![0.5, 0.2, 0.1, 0.3, 0.7, 0.6, 0.4, 0.9, 0.4, 0.9, 0.1, 0.2];
    let biases = vec![0.7, 0.2, 0.1, 0.9, 0.8];

    let network = Network::new(layer_sizes, weights, biases);

    let output = network.feed_forward(vec![5.0, 1.0]);

    black_box(output);
}

fn small_entire() {
    let layer_sizes = vec![2, 3, 2];
    let weights = vec![0.5, 0.2, 0.1, 0.3, 0.7, 0.6, 0.4, 0.9, 0.4, 0.9, 0.1, 0.2];
    let biases = vec![0.7, 0.2, 0.1, 0.9, 0.8];

    let network = Network::new(layer_sizes, weights, biases);

    let output = network.feed_forward_entire(vec![5.0, 1.0]);
    black_box(output);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("forward small entire", |b| b.iter(small_entire));
    c.bench_function("forward small", |b| b.iter(small));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
