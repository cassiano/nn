import { ActivationFunctionType } from './layer.ts'

/** A 2D array of numbers, shaped like the weight matrices (w) and the batch. */
export type NumericMatrix = number[][]

/** A 1D array of numbers: activations (a), pre-activations (z), biases (b), labels (y), images. */
export type NumericVector = number[]

/**
 * A dataset of images and their digit labels.
 * @property inputs - one flattened image per row
 * @property labels - the digit each image depicts
 */
export type TrainingData = {
  inputs: NumericMatrix
  labels: NumericVector
}

/**
 * A layer described up front, before the network exists. The input layer omits
 * σ, having no activation to apply.
 * @property name - human-readable label, used only in debug output
 * @property size - number of neurons
 * @property σ - activation function applied to this layer's pre-activations
 */
export type InitialLayerData = {
  name: string
  size: number
  σ?: ActivationFunctionType
}

/**
 * How the cost changes with one layer's parameters, shaped like the parameters
 * themselves so it can be applied to them directly.
 * @property 𝓁 - the layer's index (1-based; the input layer has no parameters)
 * @property w - ∂C/∂w(𝓁)
 * @property b - ∂C/∂b(𝓁)
 */
export type GradientLayer = {
  𝓁: number
  w: NumericMatrix
  b: NumericVector
}

/**
 * The gradient of every trainable layer, in the order
 * {@link Network.calculateGradient} produces them: output layer first.
 */
export type Gradient = GradientLayer[]
