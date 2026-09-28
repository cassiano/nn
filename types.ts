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
 * @property name - Human-readable label, used only for debugging output
 * @property size - Number of neurons in the layer
 * @property σ - Activation function applied to this layer's pre-activations.
 * Omitted for the input layer, which applies no activation.
 */
export type InitialLayerData = {
  name: string
  size: number
  σ?: ActivationFunctionType // Activation function (greek letter sigma)
}

/**
 * The gradient of the cost function with respect to one layer's parameters,
 * shaped like the parameters themselves so it can be applied directly.
 * @property 𝓁 - Index of the layer these partials belong to (1-based; 0 is the
 * input layer, which has no parameters and therefore no gradient)
 * @property w - ∂C/∂w(𝓁), a [size][previous layer size] matrix laid out in the
 * same row/column order as the layer's own weight matrix
 * @property b - ∂C/∂b(𝓁), a [size] vector aligned with the layer's biases
 */
export type LayerGradient = {
  𝓁: number // [1..𝐋]
  w: NumericMatrix
  b: NumericVector
}

/**
 * Every trainable layer's {@link LayerGradient}, ordered from the output layer
 * (𝓁 = 𝐋) backwards to the first hidden layer (𝓁 = 1), matching the order in
 * which {@link Network.calculateGradient} walks the layers.
 *
 * One entry per layer that has parameters, so its length is the number of
 * trainable layers (the output layer's index, 𝐋) rather than the number of
 * parameters. {@link Network.calculateAverageGradient} relies on that fixed
 * length and order when it pairs up a batch.
 */
export type Gradient = LayerGradient[]
