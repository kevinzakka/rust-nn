#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate npy_derive;
extern crate npy;

use rand::thread_rng;
use rand::distributions::{StandardNormal};
use rand::seq::SliceRandom;
use ndarray::prelude::*;
use ndarray::{Axis, stack};
use ndarray_rand::{RandomExt, F32};

const HIDDEN_LAYER: usize = 100;
const NUM_CLASSES: usize = 3;
const NUM_DIM: usize = 2;
const NUM_SAMPLES_PER_CLASS: usize = 100;
const NUM_ITERS: u32 = 10_000;
const L2_REG: f32 = 1e-3;
const LEARNING_RATE: f32 = 1e-0;
const DUMP: bool = true;

fn generate_data(n: usize, d: usize, k: usize) -> (Array2<f32>, Array1<u8>) {
    let mut data = Array2::<f32>::zeros((n*k, d));
    let mut label = Array1::<u8>::zeros(n*k);

    // generate
    for i in 0..k {
        let r = Array::linspace(0., 1., n);
        let noise = 0.2 * Array::random(n, F32(StandardNormal));
        let t = Array::linspace((i * 4) as f32, ((i+1)*4) as f32, n) + noise;
        let x = (&t.mapv(f32::cos) * &r).insert_axis(Axis(1));
        let y = (&t.mapv(f32::sin) * &r).insert_axis(Axis(1));
        data.slice_mut(s![i*n..(i+1)*n, ..]).assign(&stack(Axis(1), &[x.view(), y.view()]).unwrap());
        let lbl = Array1::<u8>::from_elem(n, i as u8);
        label.slice_mut(s![i*n..(i+1)*n]).assign(&lbl);
    }

    // shuffle
    let mut indices: Vec<usize> = (0..n*k).collect();
    indices.shuffle(&mut thread_rng());
    let mut X = Array2::<f32>::zeros((n*k, d));
    let mut y = Array1::<u8>::zeros(n*k);
    for (iter, elem) in indices.iter().enumerate() {
        X.slice_mut(s![iter, ..]).assign(&data.row(*elem));
        y[iter] = label[*elem];
    }

    (X, y)
}

fn cross_entropy(probas: Array2<f32>, labels: Array1<u8>) -> f32 {
    let mut loss = 0.;
    for i in 0..probas.rows() {
        loss -= f32::ln(probas.row(i)[labels[i] as usize]);
    }
    (loss / probas.rows() as f32)
}

fn main() {
    let (X, y) = generate_data(NUM_SAMPLES_PER_CLASS, NUM_DIM, NUM_CLASSES);
    let num_samples = X.rows();
    
    if DUMP {
        npy::to_file("data/X.npy", X.iter().map(|el| *el)).unwrap();
        npy::to_file("data/y.npy", y.iter().map(|el| *el)).unwrap();
    }

    let mut W1 = 0.01 * Array::random((NUM_DIM, HIDDEN_LAYER), F32(StandardNormal));
    let mut b1 = Array2::<f32>::zeros((1, HIDDEN_LAYER));
    let mut W2 = 0.01 * Array::random((HIDDEN_LAYER, NUM_CLASSES), F32(StandardNormal));
    let mut b2 = Array2::<f32>::zeros((1, NUM_CLASSES));

    for i in 0..NUM_ITERS {
        // forward pass to compute class scores
        let mut hidden_layer = X.dot(&W1) + &b1;
        hidden_layer.mapv_inplace(|x| if x < 0. { 0. } else { x });  // ReLU
        let scores = hidden_layer.dot(&W2) + &b2;

        // compute accuracy
        let mut predicted = Array1::<f32>::zeros(num_samples);
        for j in 0..num_samples {
            let row_scores = scores.row(j);
            let mut max_score_idx = 0;
            for k in 1..NUM_CLASSES {
                if row_scores[k] > row_scores[max_score_idx] {
                    max_score_idx = k;
                }
            }
            if max_score_idx == y[j] as usize {
                predicted[j] = 1.;
            }
        }
        let acc = predicted.sum() / num_samples as f32;

        // convert to class probas
        let scores_exp = scores.mapv(f32::exp);
        let probas = &scores_exp / &scores_exp.sum_axis(Axis(1)).insert_axis(Axis(1));

        // compute loss
        let mut loss = cross_entropy(probas.clone(), y.clone());
        let reg_loss = (0.5 * L2_REG * (&W1*&W1).sum()) + (0.5 * L2_REG * (&W2*&W2).sum());
        loss += reg_loss;

        if i % 1000 == 0 {
            println!("loss: {} - acc: {}", loss, acc);
        }

        // backprop gradient on scores
        let mut dscores = probas.clone();
        for (idx, lbl) in y.indexed_iter() {
            dscores.row_mut(idx)[*lbl as usize] -= 1.;
        }
        dscores.mapv_inplace(|x| x / (num_samples as f32));

        // backprop gradient to params
        let mut dW2 = hidden_layer.t().dot(&dscores);
        let db2 = dscores.sum_axis(Axis(0)).insert_axis(Axis(0));
        let mut dhidden = dscores.dot(&W2.t());
        for (idx, elem) in hidden_layer.indexed_iter() {
            if *elem <= 0. {
                dhidden[idx] = 0.;
            }
        }
        let mut dW1 = X.t().dot(&dhidden);
        let db1 = dhidden.sum_axis(Axis(0)).insert_axis(Axis(0));

        // add regularization gradients
        dW1 = dW1 + W1.mapv(|x| L2_REG * x);
        dW2 = dW2 + W2.mapv(|x| L2_REG * x);

        // update
        W1 = W1 - dW1.mapv(|x| LEARNING_RATE * x);
        b1 = b1 - db1.mapv(|x| LEARNING_RATE * x);
        W2 = W2 - dW2.mapv(|x| LEARNING_RATE * x);
        b2 = b2 - db2.mapv(|x| LEARNING_RATE * x);

        if (DUMP) && i == (NUM_ITERS - 1){
            npy::to_file("data/W1.npy", W1.iter().map(|el| *el)).unwrap();
            npy::to_file("data/W2.npy", W2.iter().map(|el| *el)).unwrap();
            npy::to_file("data/b1.npy", b1.iter().map(|el| *el)).unwrap();
            npy::to_file("data/b2.npy", b2.iter().map(|el| *el)).unwrap();
        }
    }
}
