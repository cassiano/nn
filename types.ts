/**
 * Training dataset containing input features and corresponding labels.
 * @property inputs - 2D array where each row is a flattened image (784 values)
 * @property labels - 1D array of digit labels (0-9)
 */
export type TrainingData = {
  inputs: number[][]
  labels: number[]
}

export type ActivationFnType = 'sigmoid' | 'relu' | 'tanh' | 'softmax'

export type Layer = {
  name: string
  size: number
  w?: number[][] // Weights matrix for the layer (size x previous layer size)
  b?: number[] // Bias vector for the layer (size)
  z?: number[] // Pre-activation values (size). z(𝓁) = w(𝓁) * a(𝓁-1) + b(𝓁)
  a: number[] // Activation values (size)
  σ?: ActivationFnType
}

export type Network = {
  layers: Layer[]
  y: number[] // Output values (size)
}
