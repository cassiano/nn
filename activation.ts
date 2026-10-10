import { NumericVector } from './types.ts'

// Activation functions and their derivatives, applied by the layers and by
// backpropagation. Everything is element-wise except softmax, which normalizes
// across the whole layer. Derivatives take the pre-activation z, which the
// activation value alone does not always determine.

/**
 * ReLU: max(0, x). The usual choice for hidden layers.
 *
 * @param x The value to rectify.
 * @returns x if positive, 0 otherwise.
 */
export function relu(x: number): number {
  return Math.max(0, x)
}

/**
 * ReLU': 1 for x > 0, 0 otherwise.
 *
 * @param x The pre-activation value to differentiate at.
 * @returns 1 for x > 0, 0 otherwise.
 */
export function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0
}

/**
 * Sigmoid: 1 / (1 + e^(-x)), mapping to (0, 1). Branches on the sign to stay
 * stable at extremes.
 *
 * @param x The value to squash.
 * @returns A value in (0, 1).
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
 * Sigmoid': s(1 - s) with s = sigmoid(x). Peaks at 0.25, so it flattens gradients.
 *
 * @param x The pre-activation value to differentiate at.
 * @returns s(1 - s) with s = sigmoid(x), in (0, 0.25].
 */
export function sigmoidDerivative(x: number): number {
  const s = sigmoid(x)

  return s * (1 - s)
}

/**
 * Tanh: zero-centred, mapping to (-1, 1).
 *
 * @param x The value to squash.
 * @returns A value in (-1, 1).
 */
export function tanh(x: number): number {
  return Math.tanh(x)
}

/**
 * Tanh': 1 - tanh(x)².
 *
 * @param x The pre-activation value to differentiate at.
 * @returns 1 - tanh(x)², in (0, 1].
 */
export function tanhDerivative(x: number): number {
  const t = tanh(x)

  return 1 - t * t
}

/**
 * Softmax: turns logits into a probability distribution over the layer.
 *
 * The max logit is subtracted first — it cancels out of the ratio but keeps
 * `Math.exp` from overflowing.
 *
 * @param logits The layer's pre-activation values, one per neuron.
 * @param temperature A scaling factor for the logits, where 1 is normal. Higher
 * temperatures flatten the distribution, lower ones sharpen it.
 * @returns One probability per neuron, summing to 1.
 */
export function softmax(logits: NumericVector, temperature = 1): NumericVector {
  const max = Math.max(...logits)
  const exps = logits.map(v => Math.exp((v - max) / temperature))
  const sum = exps.reduce((a, b) => a + b, 0)

  return exps.map(e => e / sum)
}
