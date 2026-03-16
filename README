# mini_torch

mini_torch is a minimal deep learning framework built from scratch to understand the core mechanics behind modern machine learning libraries.

The project implements a reverse-mode automatic differentiation engine and builds a small neural network library on top of it. The goal is to explore how frameworks like PyTorch compute gradients and train neural networks internally.

This project is inspired by the autograd lectures by Andrej Karpathy and focuses on learning deep learning systems by implementing them from first principles.

---

## Why This Project Exists

Modern deep learning libraries hide a lot of complexity behind high-level APIs. While this makes them easy to use, it can make it difficult to understand how they actually work internally.

mini_torch aims to demystify these systems by implementing:

- automatic differentiation  
- computational graphs  
- gradient propagation  
- neural network training  

from scratch.

By building these components manually, this project provides insight into the foundations of machine learning frameworks.

---

## Features

### Autograd Engine
- Reverse-mode automatic differentiation
- Dynamic computational graph construction
- Topological sorting for backpropagation
- Automatic gradient accumulation

### Mathematical Operations
- Addition
- Multiplication
- Power
- Division
- Exponential
- Logarithm

### Neural Network Components
- Neuron abstraction
- Fully connected layers
- Multi-layer perceptron (MLP)

### Training Pipeline
- Forward pass
- Loss computation
- Backpropagation
- Gradient descent parameter updates

### Example
- Training a neural network to solve the XOR classification problem

---

## Project Structure

```
mini_torch
│
├── core
│   └── autograd engine
│
├── nn
│   ├── neuron
│   ├── layer
│   └── mlp
│
├── examples
│   └── xor training
│
├── LICENSE
└── README.md
```

---

## How It Works

mini_torch represents computations as a **computational graph**.

Each value in the graph stores:
- its numerical value
- its gradient
- references to the values that produced it
- a backward function that describes how gradients propagate.

During training:

1. A forward pass builds the computation graph  
2. A loss value is computed  
3. Backpropagation traverses the graph in reverse order  
4. Gradients are propagated to every parameter  
5. Parameters are updated using gradient descent  

---

## Learning Goals

This project helps understand:

- reverse-mode automatic differentiation  
- computational graph construction  
- gradient propagation  
- neural network architecture  
- training loops in deep learning frameworks  

These concepts are fundamental to libraries such as PyTorch and TensorFlow.

---

## Future Improvements (mini_torch v2)

Planned extensions include:

- Tensor support (vector and matrix operations)
- Matrix multiplication
- Additional optimizers (Momentum, Adam)
- Cross-entropy loss
- Additional activation functions
- Training on real datasets (e.g., MNIST)

---

## License

This project is licensed under the MIT License.