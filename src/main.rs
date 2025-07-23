use machine_learning_lib::{
    init_cuda,
    layers::{
        self,
        activation::{ActivationLayer, RELU, SIGMOID},
        conv_2d::Conv2DLayer,
        dense::DenseLayer,
        optimizer::StochasticGradientDescent,
    },
};
use rand::seq::SliceRandom;

pub mod mnist_loader;

fn main() {
    let _ctx = init_cuda().expect("Failed to initialize CUDA");

    let mut network = match create_network() {
        Ok(net) => net,
        Err(e) => {
            eprintln!("Failed to create network: {}", e);
            return;
        }
    };

    let loader = mnist_loader::MnistDataloader::new(
        "./MNIST/train-images.idx3-ubyte".to_string(),
        "./MNIST/train-labels.idx1-ubyte".to_string(),
        "./MNIST/t10k-images.idx3-ubyte".to_string(),
        "./MNIST/t10k-labels.idx1-ubyte".to_string(),
    );
    let data = loader.load_data();

    let optimizer: Box<dyn machine_learning_lib::layers::optimizer::Optimizer> =
        Box::new(StochasticGradientDescent::new(0.025));

    match data {
        Ok(((train_images, train_labels), (test_images, test_labels))) => {
            // Randomly rearrange the training data
            let mut rng = rand::rng();
            let mut indices: Vec<usize> = (0..train_images.len()).collect();
            indices.shuffle(&mut rng);
            let train_images: Vec<_> = indices.iter().map(|&i| train_images[i].clone()).collect();
            let train_labels: Vec<_> = indices.iter().map(|&i| train_labels[i]).collect();

            // Convert training data to Vec<Vec<f32>>
            let train_x: Vec<Vec<f32>> = train_images
                .iter()
                .map(|img| img.iter().map(|&v| v as f32 / 255.0).collect())
                .collect();
            let train_y: Vec<u8> = train_labels.clone();

            // Mini-batch training using neural_network methods
            let batch_size = 32;
            let epochs = 15;
            for epoch in 0..epochs {
                for batch_start in (0..train_x.len()).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(train_x.len());
                    let mut cost_sum = 0.0;
                    for (x, &y) in train_x[batch_start..batch_end]
                        .iter()
                        .zip(&train_y[batch_start..batch_end])
                    {
                        let mut target = vec![0.0f32; 10];
                        target[y as usize] = 1.0;
                        let output = network.infer(&x).unwrap();

                        network.back_propagate(&target).unwrap();
                        let cost = output
                            .iter()
                            .zip(target.iter())
                            .map(|(o, t)| (o - t).powi(2))
                            .sum::<f32>()
                            .sqrt();
                        cost_sum += cost;
                    }
                    network.optimize(&optimizer).unwrap();
                    println!(
                        "Processed batch {}/{}",
                        batch_start / batch_size + 1,
                        (train_x.len() + batch_size - 1) / batch_size
                    );
                    let average_cost = cost_sum / batch_size as f32;
                    println!("Epoch {}: Batch average cost: {:.2}", epoch, average_cost);
                }
            }

            let test_x: Vec<Vec<f32>> = test_images
                .iter()
                .map(|img| img.iter().map(|&v| v as f32 / 255.0).collect())
                .collect();
            let test_y: Vec<u8> = test_labels.clone();

            let mut correct = 0;
            for (img, &label) in test_x.iter().zip(test_y.iter()) {
                let output = network.infer(&img).unwrap();
                let prediction = nalgebra::DVector::from_vec(output).argmax().0 as u8;
                if prediction == label {
                    correct += 1;
                }
            }
            let accuracy = correct as f32 / test_x.len() as f32;

            println!("Training complete. Test accuracy: {:.2}%", accuracy * 100.0)
        }
        Err(e) => println!("Error loading data: {}", e),
    }
}

pub fn create_network() -> Result<layers::LayerWrapper, Box<dyn std::error::Error>> {
    let network = layers::LayerWrapper::new(layers::network::NeuralNetwork::boxed(vec![
        Conv2DLayer::boxed(1, 28, 28, 5, 5, 32, layers::dense::xavier_initializer),
        ActivationLayer::boxed(RELU, 24 * 24 * 32),
        Conv2DLayer::boxed(32, 24, 24, 5, 5, 20, layers::dense::xavier_initializer),
        ActivationLayer::boxed(RELU, 20 * 20 * 20),
        Conv2DLayer::boxed(20, 20, 20, 5, 5, 16, layers::dense::xavier_initializer),
        ActivationLayer::boxed(RELU, 16 * 16 * 16),
        DenseLayer::boxed(16 * 16 * 16, 256, layers::dense::xavier_initializer),
        ActivationLayer::boxed(RELU, 256),
        DenseLayer::boxed(256, 128, layers::dense::xavier_initializer),
        ActivationLayer::boxed(RELU, 128),
        DenseLayer::boxed(128, 10, layers::dense::he_initializer),
        ActivationLayer::boxed(SIGMOID, 10),
    ]))?;
    Ok(network)
}
