import { ActivationFunctionType } from './layer.ts'

/**
 * A 2D array of numbers.
 * Used for weight matrices (w), as `[size][previous layer size]`, and for
 * batched training inputs.
 */
export type NumericMatrix = number[][]

/**
 * A 1D array of numbers.
 * Used for activations (a), pre-activations (z), biases (b), one-hot label
 * vectors (y) and flattened images (784 pixel values).
 */
export type NumericVector = number[]

/**
 * Training dataset containing input features and corresponding labels.
 * @property inputs - 2D array where each row is a flattened image (784 values)
 * @property labels - 1D array of digit labels (0-9)
 */
export type TrainingData = {
  inputs: NumericMatrix
  labels: NumericVector
}

/**
 * High-level description of a layer passed to the Network constructor.
 * `size` is the number of neurons; `σ` (optional) is the activation function
 * applied to that layer. The input layer omits σ since it has no weights.
 */
export type InitialLayerData = {
  name: string
  size: number
  σ?: ActivationFunctionType // Activation function (greek letter sigma)
}
