///////////////////////
// AI-generated code //
///////////////////////

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
 *
 * @example
 * relu(-2) // 0
 * relu(3) // 3
 */
export function relu(x: number): number {
  return Math.max(0, x)
}

/**
 * ReLU': 1 for x > 0, 0 otherwise.
 *
 * @param x The pre-activation value to differentiate at.
 * @returns 1 for x > 0, 0 otherwise.
 *
 * @example
 * reluDerivative(-0.5) // 0
 * reluDerivative(0.5) // 1
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
 *
 * @example
 * sigmoid(0) // 0.5
 * sigmoid(2) // ≈ 0.8808
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
 *
 * @example
 * sigmoidDerivative(0) // 0.25
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
 *
 * @example
 * tanh(0) // 0
 * tanh(1) // ≈ 0.7616
 */
export function tanh(x: number): number {
  return Math.tanh(x)
}

/**
 * Tanh': 1 - tanh(x)².
 *
 * @param x The pre-activation value to differentiate at.
 * @returns 1 - tanh(x)², in (0, 1].
 *
 * @example
 * tanhDerivative(0) // 1
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
 * A temperature of 0 is not representable by the formula but is the limit in
 * which the distribution becomes one-hot on the largest logit(s), so it is
 * handled explicitly; ties split the mass evenly.
 *
 * @param logits The layer's pre-activation values, one per neuron.
 * @param temperature A scaling factor for the logits, where 1 is normal. Higher
 * temperatures flatten the distribution, lower ones sharpen it. Must be
 * non-negative; 0 yields a one-hot argmax distribution.
 * @returns One probability per neuron, summing to 1.
 *
 * @example
 * softmax([1, 2, 3]) // ≈ [0.0900, 0.2447, 0.6652]
 * softmax([1, 1], 0) // [0.5, 0.5]
 */
export function softmax(logits: NumericVector, temperature = 1): NumericVector {
  if (temperature < 0)
    throw new Error('Softmax temperature must be non-negative')

  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]

  if (temperature === 0) {
    let winners = 0

    for (let i = 0; i < logits.length; i++) if (logits[i] === max) winners++

    const share = 1 / winners
    const result: NumericVector = new Array(logits.length)

    for (let i = 0; i < logits.length; i++)
      result[i] = logits[i] === max ? share : 0

    return result
  }

  const exps: NumericVector = new Array(logits.length)
  let sum = 0

  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp((logits[i] - max) / temperature)
    exps[i] = e
    sum += e
  }

  const result: NumericVector = new Array(logits.length)

  for (let i = 0; i < logits.length; i++) result[i] = exps[i] / sum

  return result
}
