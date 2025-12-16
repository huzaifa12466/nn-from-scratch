# Neural Network from Scratch – MNIST Classification

## Project Overview

This project implements a simple **feedforward neural network from scratch** using **NumPy** to classify handwritten digits from the **MNIST dataset**.

The network consists of an input layer, one hidden layer, and an output layer. The main goal is to understand and implement fundamental deep learning concepts **without using high-level frameworks** like TensorFlow or PyTorch.

---

## Features

* **Network Architecture:**

  * Input layer: 784 neurons (28x28 flattened images)
  * Hidden layer: 64 neurons, **ReLU** activation
  * Output layer: 10 neurons, **Softmax** activation

* **Forward propagation:** Linear transformation + activation (ReLU for hidden layers, Softmax for output layer)

* **Loss function:** Cross-entropy loss

* **Backpropagation:** Gradients computed manually using the chain rule

* **Parameter update:** Gradient descent with configurable learning rate

* **One-hot encoding:** Converts labels into one-hot vectors for classification

* **Performance tracking:** Plots for training and test loss, training loss vs accuracy, and final accuracy

---

## Files

* **`nn_from_scratch_mnist.ipynb`** – Main notebook containing the complete code from data preprocessing to training and evaluation

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/huzaifa12466/nn-from-scratch.git
```

2. Install required packages manually:

```bash
pip install numpy matplotlib pandas
```

---

## Usage

1. Load and preprocess the MNIST dataset (flatten images and normalize).
2. One-hot encode the labels:

```python
Y_train_one_hot = one_hot_encode(y_train, num_classes)
Y_test_one_hot = one_hot_encode(y_test, num_classes)
```

3. Initialize parameters:

```python
parameters = initialize_parameters(layer_dims)
```

4. Train the model:

```python
parameters, train_losses, test_losses, train_acc, test_acc = train_model(
    X_train, Y_train_one_hot, X_test, Y_test_one_hot,
    layer_dims, learning_rate=0.01, epochs=500, print_every=10
)
```

5. Evaluate and visualize results:

* Training and test loss vs epochs
* Loss vs accuracy plot (training)
* Final accuracy bar chart (train vs test)

---

## Results

* After **500 epochs**, the model achieved the following:

  * **Train Loss:** 0.6400
  * **Train Accuracy:** 84.69%
  * **Test Loss:** 0.6162
  * **Test Accuracy:** 85.20%

* **Plots included:**

  * Training and test loss over epochs
  * Loss vs Accuracy (Training)
  * Final accuracy bar chart

This demonstrates that the network successfully learned to classify handwritten digits with good accuracy using a simple one-hidden-layer architecture.

---

## Functions

* `initialize_parameters(layer_dims)` – Initializes weights and biases using **He initialization**.
* `relu(Z)` – ReLU activation function for hidden layers.
* `softmax(Z)` – Softmax activation for output layer.
* `forward_propagation(X, parameters)` – Computes activations layer by layer.
* `compute_loss(A2, Y)` – Computes cross-entropy loss.
* `compute_accuracy(A2, Y)` – Computes classification accuracy.
* `update_parameters(parameters, grads, learning_rate)` – Updates parameters using gradient descent.
* `one_hot_encode(labels, num_classes)` – Converts integer labels to one-hot vectors.

---

## Notes

Dataset should be normalized before training for better convergence.

Learning rate and hidden layer size can be tuned to improve performance.

Since this model is implemented from scratch with NumPy, training is slower than using frameworks like PyTorch or TensorFlow.

This project demonstrates the complete neural network workflow: parameter initialization → forward propagation → loss computation → backpropagation → parameter updates → evaluation.

