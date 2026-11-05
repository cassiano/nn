import { CostFunctionType, NumericVector } from './types.ts'

// Cost functions compare the target y with the output layer's activations a.
// Each one comes with the derivative ∂C/∂a that starts backpropagation. Both
// costs are summed over the neurons with no 1/n factor, which only rescales the
// gradient and keeps the learning rate's meaning unchanged.

/**
 * Squared error: C = Σ(yᵢ - aᵢ)².
 *
 * @param target The expected output y.
 * @param output The output layer's activations a.
 * @returns The summed squared error.
 *
 * @example
 * meanSquaredError([1, 0], [0.75, 0.1]) // 0.0725
 */
export function meanSquaredError(
  target: NumericVector,
  output: NumericVector,
): number {
  let cost = 0

  for (let i = 0; i < target.length; i++) {
    const difference = target[i] - output[i]

    cost += difference * difference
  }

  return cost
}

/**
 * Squared error's derivative with respect to the output activations:
 * ∂C/∂aᵢ = 2(aᵢ - yᵢ).
 *
 * @param target The expected output y.
 * @param output The output layer's activations a.
 * @returns One partial derivative per output neuron.
 *
 * @example
 * meanSquaredErrorDerivative([1, 0], [0.75, 0.1]) // [-0.5, 0.2]
 */
export function meanSquaredErrorDerivative(
  target: NumericVector,
  output: NumericVector,
): NumericVector {
  return output.map((activation, i) => 2 * (activation - target[i]))
}

/**
 * Cross-entropy: C = -Σ yᵢ·ln(aᵢ). With one-hot targets only the true class
 * contributes, but soft targets are supported as well.
 *
 * @param target The expected output y.
 * @param output The output layer's activations a, read as probabilities.
 * @returns The cross-entropy.
 *
 * @example
 * crossEntropy([0, 1], [0.2, 0.8]) // ≈ 0.2231
 */
export function crossEntropy(
  target: NumericVector,
  output: NumericVector,
): number {
  let cost = 0

  for (let i = 0; i < target.length; i++)
    if (target[i] !== 0) cost -= target[i] * Math.log(output[i])

  return cost
}

/**
 * Cross-entropy's derivative with respect to the output activations:
 * ∂C/∂aᵢ = -yᵢ/aᵢ.
 *
 * @param target The expected output y.
 * @param output The output layer's activations a, read as probabilities.
 * @returns One partial derivative per output neuron.
 *
 * @example
 * crossEntropyDerivative([0, 1], [0.2, 0.8]) // [0, -1.25]
 */
export function crossEntropyDerivative(
  target: NumericVector,
  output: NumericVector,
): NumericVector {
  return output.map((activation, i) =>
    target[i] === 0 ? 0 : -target[i] / activation,
  )
}

/**
 * Evaluates the selected cost function.
 *
 * @param type Which cost to evaluate.
 * @param target The expected output y.
 * @param output The output layer's activations a.
 * @returns The cost for this sample.
 */
export function costValue(
  type: CostFunctionType,
  target: NumericVector,
  output: NumericVector,
): number {
  switch (type) {
    case 'mse':
      return meanSquaredError(target, output)
    case 'cross-entropy':
      return crossEntropy(target, output)
    default: {
      const _exhaustiveCheck: never = type
      return _exhaustiveCheck
    }
  }
}

/**
 * Evaluates the selected cost function's derivative ∂C/∂a.
 *
 * @param type Which cost to differentiate.
 * @param target The expected output y.
 * @param output The output layer's activations a.
 * @returns One partial derivative per output neuron.
 */
export function costDerivative(
  type: CostFunctionType,
  target: NumericVector,
  output: NumericVector,
): NumericVector {
  switch (type) {
    case 'mse':
      return meanSquaredErrorDerivative(target, output)
    case 'cross-entropy':
      return crossEntropyDerivative(target, output)
    default: {
      const _exhaustiveCheck: never = type
      return _exhaustiveCheck
    }
  }
}
