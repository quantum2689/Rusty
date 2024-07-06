use std::vec::Vec;

struct Perceptron {
    weights: Vec<f32>,
    bias: f32,
    learning_rate: f32,
}

impl Perceptron {
    // Initialize the perceptron with a given number of inputs
    fn new(num_inputs: usize, learning_rate: f32) -> Self {
        Self {
            weights: vec![0.0; num_inputs],
            bias: 0.0,
            learning_rate,
        }
    }

    // Activation function (step function)
    fn activation_function(&self, sum: f32) -> i32 {
        if sum >= 0.0 { 1 } else { 0 }
    }

    // Predict the output for a given input
    fn predict(&self, inputs: &[f32]) -> i32 {
        let sum: f32 = self.weights.iter().zip(inputs.iter()).map(|(w, i)| w * i).sum::<f32>() + self.bias;
        self.activation_function(sum)
    }

    // Train the perceptron with given training data
    fn train(&mut self, training_data: &[(Vec<f32>, i32)]) {
        for (inputs, target) in training_data.iter() {
            let prediction = self.predict(inputs);
            let error = target - prediction;
            // Update weights and bias
            for i in 0..self.weights.len() {
                self.weights[i] += self.learning_rate * error as f32 * inputs[i];
            }
            self.bias += self.learning_rate * error as f32;
        }
    }
}

fn main() {
    // Example training data (AND logic gate)
    // Each tuple consists of input vector and the target output
    let training_data = vec![
        (vec![0.0, 0.0], 0),
        (vec![0.0, 1.0], 0),
        (vec![1.0, 0.0], 0),
        (vec![1.0, 1.0], 1),
    ];

    // Initialize the perceptron
    let mut perceptron = Perceptron::new(2, 0.1);

    // Train the perceptron
    for _ in 0..10 { // Train for 10 epochs
        perceptron.train(&training_data);
    }

    // Test the perceptron
    println!("Prediction for [0, 0]: {}", perceptron.predict(&[0.0, 0.0]));
    println!("Prediction for [0, 1]: {}", perceptron.predict(&[0.0, 1.0]));
    println!("Prediction for [1, 0]: {}", perceptron.predict(&[1.0, 0.0]));
    println!("Prediction for [1, 1]: {}", perceptron.predict(&[1.0, 1.0]));
}
