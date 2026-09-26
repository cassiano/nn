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
 *
 * @param x The input value.
 * @returns `x` when `x > 0`, otherwise 0. Note 0 itself returns 0.
 */
export function relu(x: number): number {
  return Math.max(0, x)
}

/**
 * Derivative of ReLU: 1 for positive inputs, 0 otherwise
 *
 * @param x The pre-activation value z at which to evaluate the derivative.
 * @returns 1 when `x > 0`, otherwise 0. The derivative at exactly 0 is
 * defined here as 0.
 */
export function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0
}

/**
 * Sigmoid: f(x) = 1 / (1 + e^(-x))
 * - Maps any real number to (0, 1)
 * - Used for binary classification or as an alternative hidden layer activation
 * - Numerically stable: handles positive and negative inputs separately to avoid overflow
 * - Derivative: σ(x) * (1 - σ(x))
 *
 * @param x The input value; any finite number is accepted.
 * @returns A value in the open interval (0, 1): above 0.5 for positive `x`,
 * exactly 0.5 at 0, below 0.5 for negative `x`. Never returns exactly 0 or 1.
 */
export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x)

    return 1 / (1 + z)
  }

  const z = Math.exp(x)

  return z / (1 + z)
}

/**
 * Derivative of sigmoid: s * (1 - s) where s = sigmoid(x)
 *
 * @param x The pre-activation value z at which to evaluate the derivative.
 * @returns A value in [0, 0.25], peaking at 0.25 where x = 0 and approaching 0
 * as |x| grows. Requires the pre-activation rather than the activation,
 * because it is not recoverable from the output alone.
 */
export function sigmoidDerivative(x: number): number {
  const s = sigmoid(x)

  return s * (1 - s)
}

/**
 * Tanh (Hyperbolic Tangent): f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * - Maps any real number to (-1, 1)
 * - Zero-centered, which can help with convergence compared to sigmoid
 * - Derivative: 1 - tanh(x)^2
 *
 * @param x The input value; any finite number is accepted.
 * @returns A value in the open interval (-1, 1): 0 at 0, approaching 1 as `x`
 * grows and -1 as `x` shrinks.
 */
export function tanh(x: number): number {
  return Math.tanh(x)
}

/**
 * Derivative of tanh: 1 - tanh(x)^2
 *
 * @param x The pre-activation value z at which to evaluate the derivative.
 * @returns A value in [0, 1], equal to 1 at x = 0 and approaching 0 as |x|
 * grows.
 */
export function tanhDerivative(x: number): number {
  const t = tanh(x)

  return 1 - t * t
}

/**
 * Softmax: converts a vector of logits into a probability distribution
 * - Used in the output layer for multi-class classification
 * - Numerically stable: subtracts the max logit before exponentiation to prevent overflow
 *
 * Formula: softmax(z_i) = e^(z_i) / Σ e^(z_j)
 *
 * @param logits The pre-activation values, one per class.
 * @returns A vector of the same length whose entries are all strictly positive
 * and sum to 1. The max logit is subtracted first, which cancels out of the
 * ratio but prevents `Math.exp` from overflowing. An empty input yields `[]`,
 * since the sum then reduces to 0 over no terms.
 */
export function softmax(logits: number[]): number[] {

  const max = Math.max(...logits)
  const exps = logits.map(v => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)

  return exps.map(e => e / sum)
}
