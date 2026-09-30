import { NumericVector } from './types.ts'

// Activation functions and their derivatives, applied by the layers and by
// backpropagation. Everything is element-wise except softmax, which normalizes
// across the whole layer. Derivatives take the pre-activation z, which the
// activation value alone does not always determine.

/** ReLU: max(0, x). The usual choice for hidden layers. */
export function relu(x: number): number {
  return Math.max(0, x)
}

/** ReLU': 1 for x > 0, 0 otherwise. */
export function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0
}

/** Sigmoid: 1 / (1 + e^(-x)), mapping to (0, 1). Branches on the sign to stay stable at extremes. */
export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x)

    return 1 / (1 + z)
  }

  const z = Math.exp(x)

  return z / (1 + z)
}

/** Sigmoid': s(1 - s) with s = sigmoid(x). Peaks at 0.25, so it flattens gradients. */
export function sigmoidDerivative(x: number): number {
  const s = sigmoid(x)

  return s * (1 - s)
}

/** Tanh: zero-centred, mapping to (-1, 1). */
export function tanh(x: number): number {
  return Math.tanh(x)
}

/** Tanh': 1 - tanh(x)². */
export function tanhDerivative(x: number): number {
  const t = tanh(x)

  return 1 - t * t
}

/**
 * Softmax: turns logits into a probability distribution over the layer.
 *
 * The max logit is subtracted first — it cancels out of the ratio but keeps
 * `Math.exp` from overflowing.
 */
export function softmax(logits: NumericVector): NumericVector {
  const max = Math.max(...logits)
  const exps = logits.map(v => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)

  return exps.map(e => e / sum)
}
