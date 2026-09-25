import { NumericVector, NumericMatrix } from './types.ts'
import { timesForEachN, timesForEach } from './utils.ts'
import { Network } from './network.ts'
import {
  random,
  timesMap,
  timesMapN,
  fromMatrix,
  addMatrices,
  multiplyMatrices,
  toMatrix,
} from './utils.ts'
import {
  sigmoidDerivative,
  reluDerivative,
  tanhDerivative,
  relu,
  tanh,
  softmax,
  sigmoid,
} from './activation.ts'

export type ActivationFunctionType = 'sigmoid' | 'relu' | 'tanh' | 'softmax'

/**
 * A single layer of a {@link Network}: a fixed-size group of neurons.
 *
 * Stores the layer's trainable parameters and intermediate values:
 * - w: weight matrix ([size][previous layer size]), random in (-1, 1) at init
 * - b: bias vector ([size]), random in (-1, 1) at init
 * - z: pre-activation values z(𝓁) = w(𝓁)·a(𝓁-1) + b(𝓁)
 * - a: post-activation values, obtained by applying σ to z
 *
 * The first (input) layer is a special case: it only stores the raw inputs in
 * `a` and completely ignores w, b, z and σ.
 */
export class Layer {
  𝓁: number
  a: NumericVector = [] // Post-activation values ([size])
  w: NumericMatrix = [] // Weights ([size][previous layer size])
  b: NumericVector = [] // Biases ([size])
  z: NumericVector = [] // Pre-activation values ([size])

  static id = 0

  constructor(
    public network: Network,
    public name: string,
    public size: number,
    public σ?: ActivationFunctionType, // Activation function (greek letter sigma)
  ) {
    this.𝓁 = Layer.id++

    if (this.𝓁 > 0) this.initializeNetworkParameters()
  }

  get parameterCount() {
    return this.𝓁 === 0 ? 0 : this.w.length * this.w[0].length + this.b.length
  }

  get previousLayer() {
    return this.network.previousLayer(this.𝓁)
  }

  // ∂C/∂w(𝓁)(𝒿, 𝚔), ∂C/∂b(𝓁)(𝒿), 1 ≤ 𝓁 ≤ 𝐋, 𝒿: layer 𝓁, 𝚔: layer 𝓁-1
  // [
  //   ∂C/∂w(1)(0, 0), ∂C/∂w(1)(0, 1), ∂C/∂w(1)(0, 2), ..., ∂C/∂b(1)(0), ∂C/∂b(1)(1), ... // Layer 2's weights and biases partial derivatives
  //   ...
  //   ∂C/∂w(𝐋-1)(0, 0), ∂C/∂w(𝐋-1)(0, 1), ∂C/∂w(𝐋-1)(0, 2), ..., ∂C/∂b(𝐋-1)(0), ∂C/∂b(𝐋-1)(1), ... // Layer 𝐋-1's weights and biases partial derivatives
  //   ∂C/∂w(𝐋)(0, 0), ∂C/∂w(𝐋)(0, 1), ∂C/∂w(𝐋)(0, 2), ..., ∂C/∂b(𝐋)(0), ∂C/∂b(𝐋)(1), ... // Layer 𝐋's weights and biases partial derivatives
  // ]
  calculateGradient() {
    const gradient: NumericVector = []
    const derivativeOfσ = this.derivativeOfσ()
    const previousLayer = this.previousLayer // Cache it into a local variable.
    const weightRows = this.w.length // this.size
    const weightCols = this.w[0].length // previousLayer.size

    timesForEachN([weightRows, weightCols], (row, col) => {
      // ∂C/∂w
      gradient[row * weightCols + col] =
        previousLayer.a[col] * // ∂z/∂w
        derivativeOfσ(this.z[row]) * // ∂a/∂z
        (2 * (this.a[row] - this.network.y[row])) // ∂C/∂a
    })

    timesForEach(this.b.length, i => {
      // ∂C/∂b
      gradient[weightRows * weightCols + i] =
        1 * // ∂z/∂b
        derivativeOfσ(this.z[i]) * // ∂a/∂z
        (2 * (this.a[i] - this.network.y[i])) // ∂C/∂a
    })

    return gradient
  }

  calculatePostActivationValues() {
    this.calculatePreActivationValues()
    this.applyActivationFunction()
  }

  private initializeNetworkParameters() {
    this.w = timesMapN([this.size, this.previousLayer.size], () =>
      random(-1, 1),
    )
    this.b = timesMap(this.size, () => random(-1, 1))
  }

  private calculatePreActivationValues() {
    // z(𝓁) = w(𝓁) * a(𝓁-1) + b(𝓁)
    this.z = fromMatrix(
      addMatrices(
        multiplyMatrices(this.w, toMatrix(this.previousLayer.a)),
        toMatrix(this.b),
      ),
    )
  }

  private applyActivationFunction() {
    if (this.σ === undefined)
      throw new Error(`Expected activation function to be defined`)

    switch (this.σ) {
      case 'sigmoid':
        this.a = this.z.map(sigmoid)
        break
      case 'relu':
        this.a = this.z.map(relu)
        break
      case 'tanh':
        this.a = this.z.map(tanh)
        break
      case 'softmax':
        this.a = softmax(this.z)
        break
      default: {
        const exhaustiveCheck: never = this.σ
        throw exhaustiveCheck
      }
    }
  }

  private derivativeOfσ() {
    if (this.σ === undefined)
      throw new Error(`Expected activation function to be defined`)

    switch (this.σ) {
      case 'sigmoid':
        return sigmoidDerivative
      case 'relu':
        return reluDerivative
      case 'tanh':
        return tanhDerivative
      case 'softmax':
        return () => 1
      default: {
        const exhaustiveCheck: never = this.σ
        throw exhaustiveCheck
      }
    }
  }
}
