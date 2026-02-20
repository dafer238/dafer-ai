Week 07 — Neural Networks From Scratch

## Prerequisites

- **Week 00b** — the chain rule. Backpropagation is the chain rule applied repeatedly through a composition of functions.
- **Week 03** — linear regression is a single-layer neural network without activation. The extension to multiple layers and nonlinearities is the only new concept.
- **Week 01–02** — gradient descent. The training loop is identical; only the gradient computation changes.

## What this week delivers

You will implement every part of a neural network by hand: forward pass, loss, backward pass, and parameter update. After this week, PyTorch autograd (Week 09) will have no mystery — you will know exactly what it is computing.

Overview
Implement neural networks from first principles to fully understand forward/backprop, activations, initialization, and debugging.

Study
- Neurons, activation functions (ReLU, tanh, sigmoid), weight initialization
- Chain rule and backprop derivations

Practical libraries & tools
- NumPy for implementations; Matplotlib for visualizations
- Optional: JAX for automatic differentiation to test implementations

Datasets & examples
- Toy classification/regression datasets (spirals, moons, sine regression)

Exercises
1) Implement a fully connected NN
   - Forward pass, loss, backward pass, parameter update loop.

2) Gradient checking
   - Use finite differences to validate analytic gradients.

3) Initialization experiments
   - Compare Xavier/He vs small random init and track training dynamics.

Code hint
  # simple forward for single hidden layer
  h = np.dot(X, W1) + b1
  a = np.maximum(0, h)
  y_hat = np.dot(a, W2) + b2

Reading & resources
- Andrew Ng notes, CS231n backprop notes, and simple blog posts on initialization.

Deliverable
- Notebook implementing NN from scratch, gradient checks, and experiments showing initialization/activation effects.

## What comes next

- **Week 08** uses the same NumPy implementation to study training pathologies (vanishing gradients, dead neurons).
- **Week 09** re-implements this week's network in PyTorch; you verify both give identical outputs — that's how you know autograd is correct.
