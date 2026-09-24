import { NumericVector, NumericMatrix } from './types.ts'
import { timesForEachN, timesForEach } from './utils.ts'
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

export type ActivationFunctionType =
  | 'none'
  | 'sigmoid'
  | 'relu'
  | 'tanh'
  | 'softmax'

export class Layer {
  a: NumericVector = [] // Activation values [size]
  w: NumericMatrix = [] // Weights matrix for the layer [size][previous layer size]
  b: NumericVector = [] // Bias vector for the layer [size]
  z: NumericVector = [] // Pre-activation values [size]

  constructor(
    public name: string,
    public size: number,
    public σ: ActivationFunctionType = 'none',
  ) {}

  initializeParameters(previousLayer: Layer) {
    this.w = timesMapN([this.size, previousLayer.size], () => random(-1, 1))
    this.b = timesMap(this.size, () => random(-1, 1))
  }

  get parameterCount() {
    return this.w.length === 0
      ? 0
      : this.w.length * this.w[0].length + this.b.length
  }

  calculateActivationValues(previousLayer: Layer) {
    this.calculatePreActivationValues(previousLayer)
    this.applyActivationFunction()
  }

  // ∂C/∂w(𝓁)(𝒿, 𝚔), ∂C/∂b(𝓁)(𝒿), 1 ≤ 𝓁 ≤ 𝐋, 𝒿: layer 𝓁, 𝚔: layer 𝓁-1
  // [
  //   ∂C/∂w(1)(0, 0), ∂C/∂w(1)(0, 1), ∂C/∂w(1)(0, 2), ..., ∂C/∂b(1)(0), ∂C/∂b(1)(1), ... // Layer 2's weights and biases partial derivatives
  //   ...
  //   ∂C/∂w(𝐋-1)(0, 0), ∂C/∂w(𝐋-1)(0, 1), ∂C/∂w(𝐋-1)(0, 2), ..., ∂C/∂b(𝐋-1)(0), ∂C/∂b(𝐋-1)(1), ... // Layer 𝐋-1's weights and biases partial derivatives
  //   ∂C/∂w(𝐋)(0, 0), ∂C/∂w(𝐋)(0, 1), ∂C/∂w(𝐋)(0, 2), ..., ∂C/∂b(𝐋)(0), ∂C/∂b(𝐋)(1), ... // Layer 𝐋's weights and biases partial derivatives
  // ]
  calculateGradient(previousLayer: Layer, y: NumericVector) {
    const gradient: NumericVector = []

    timesForEachN(
      [this.size, previousLayer.size], // Equivalent to: [this.w.length, this.w[0].length]
      (row, col) => {
        gradient[row * previousLayer.size + col] =
          previousLayer.a[col] * // ∂z/∂w
          this.activationFunctionDerivative()(this.z[row]) * // ∂a/∂z
          (2 * (this.a[row] - y[row])) // ∂C/∂a
      },
    )

    timesForEach(this.size, i => {
      gradient[this.size * previousLayer.size + i] =
        1 * // ∂z/∂b
        this.activationFunctionDerivative()(this.z[i]) * // ∂a/∂z
        (2 * (this.a[i] - y[i])) // ∂C/∂a
    })

    return gradient
  }

  private calculatePreActivationValues(previousLayer: Layer) {
    // z(𝓁) = w(𝓁) * a(𝓁-1) + b(𝓁)
    this.z = fromMatrix(
      addMatrices(
        multiplyMatrices(this.w, toMatrix(previousLayer.a)),
        toMatrix(this.b),
      ),
    )
  }

  private applyActivationFunction() {
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
      case 'none':
        throw new Error(
          `Expected current layer to contain a valid activation function, but got '${this.σ}'`,
        )
      default: {
        const exhaustiveCheck: never = this.σ
        throw exhaustiveCheck
      }
    }
  }

  private activationFunctionDerivative() {
    switch (this.σ) {
      case 'sigmoid':
        return sigmoidDerivative
      case 'relu':
        return reluDerivative
      case 'tanh':
        return tanhDerivative
      case 'softmax':
        return () => 1
      case 'none':
        throw new Error(
          `Expected current layer to contain a valid activation function, but got '${this.σ}'`,
        )
      default: {
        const exhaustiveCheck: never = this.σ
        throw exhaustiveCheck
      }
    }
  }
}
