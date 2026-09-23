export type NumericMatrix = number[][]
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

export type ActivationFnType = 'sigmoid' | 'relu' | 'tanh' | 'softmax'

export type Layer = {
  name: string
  size: number
  w?: NumericMatrix // Weights matrix for the layer [size][previous layer size]
  b?: NumericVector // Bias vector for the layer [size]
  z?: NumericVector // Pre-activation values [size]
  a: NumericVector // Activation values [size]
  σ?: ActivationFnType
}
