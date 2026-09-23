/**
 * Activation functions and their derivatives for the neural network.
 *
 * These are pure functions applied element-wise to neuron outputs.
 * The derivative functions take the *pre-activation* value (z) as input,
 * since we need z to compute the derivative during backpropagation.
 */

/**
 * ReLU (Rectified Linear Unit): f(x) = max(0, x)
 * - Most common activation for hidden layers
 * - Outputs zero for negative inputs, passes positive values unchanged
 * - Derivative: 1 if x > 0, else 0
 */
export function relu(x: number): number {
  return x > 0 ? x : 0
}

/** Derivative of ReLU: 1 for positive inputs, 0 otherwise */
export function reluDeriv(x: number): number {
  return x > 0 ? 1 : 0
}

/**
 * Sigmoid: f(x) = 1 / (1 + e^(-x))
 * - Maps any real number to (0, 1)
 * - Used for binary classification or as an alternative hidden layer activation
 * - Numerically stable: handles positive and negative inputs separately to avoid overflow
 * - Derivative: σ(x) * (1 - σ(x))
 */
export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x)

    return 1 / (1 + z)
  }

  const z = Math.exp(x)

  return z / (1 + z)
}

/** Derivative of sigmoid: s * (1 - s) where s = sigmoid(x) */
export function sigmoidDeriv(x: number): number {
  const s = sigmoid(x)

  return s * (1 - s)
}

/**
 * Tanh (Hyperbolic Tangent): f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * - Maps any real number to (-1, 1)
 * - Zero-centered, which can help with convergence compared to sigmoid
 * - Derivative: 1 - tanh(x)^2
 */
export function tanh(x: number): number {
  return Math.tanh(x)
}

/** Derivative of tanh: 1 - tanh(x)^2 */
export function tanhDeriv(x: number): number {
  const t = Math.tanh(x)

  return 1 - t * t
}

/**
 * Softmax: converts a vector of logits into a probability distribution.
 * - Each output is in (0, 1) and all outputs sum to 1
 * - Used in the output layer for multi-class classification
 * - Numerically stable: subtracts the max logit before exponentiation to prevent overflow
 *
 * Formula: softmax(z_i) = e^(z_i) / Σ e^(z_j)
 */
export function softmax(logits: number[]): number[] {
  const max = Math.max(...logits)
  const exps = logits.map(v => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)

  return exps.map(e => e / sum)
}
