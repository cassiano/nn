import {
  random,
  timesMap,
  fromMatrix,
  addMatrices,
  multiplyMatrices,
  toMatrix,
  timesMapN,
} from './utils.ts'
import {
  sigmoid,
  relu,
  tanh,
  sigmoidDeriv,
  reluDeriv,
  tanhDeriv,
} from './activation.ts'
import { softmax } from './activation.ts'
import { ParameterizedLayer, NumericVector, Layer } from './types.ts'
import { MNIST_OUTPUT_SIZE } from './constants.ts'
import { timesForEachN, timesForEach } from './utils.ts'

export class Network {
  y: NumericVector = [] // Expected values for the current sample

  constructor(public layers: [Layer, ...ParameterizedLayer[]]) {
    if (layers[layers.length - 1].size !== MNIST_OUTPUT_SIZE)
      throw new Error(
        `Expected output layer size of ${MNIST_OUTPUT_SIZE} but got ${layers[layers.length - 1].size}`,
      )

    // Fill weights and biases of all but the input layer with random data.
    for (let i = 0; i < this.parameterizedLayers.length; i++) {
      const previousLayer =
        i === 0 ? this.inputLayer : this.parameterizedLayers[i - 1]
      const currentLayer = this.parameterizedLayers[i]

      currentLayer.w = timesMapN([currentLayer.size, previousLayer.size], () =>
        random(-1, 1),
      )

      currentLayer.b = timesMap(currentLayer.size, () => random(-1, 1))
    }
  }

  get inputLayer() {
    return this.layers[0]
  }

  get parameterizedLayers() {
    return this.layers.slice(1) as ParameterizedLayer[]
  }

  get outputLayer() {
    return this.layers[this.layers.length - 1] as ParameterizedLayer
  }

  get parameterCount() {
    return this.parameterizedLayers.reduce(
      (count, layer) =>
        count + layer.w.length * layer.w[0].length + layer.b.length,
      0,
    )
  }

  loadSample(inputs: number[], label: number) {
    if (inputs.length !== this.inputLayer.size)
      throw new Error(
        `Expected input size of ${this.inputLayer.size} but got ${inputs.length}`,
      )

    this.inputLayer.a = inputs

    this.y = timesMap(MNIST_OUTPUT_SIZE, i => (i === label ? 1 : 0))
  }

  feedForward() {
    for (let i = 0; i < this.parameterizedLayers.length; i++) {
      // console.log(`Processing layer ${i}: ${this.layers[i].name}`)

      const previousLayer =
        i === 0 ? this.inputLayer : this.parameterizedLayers[i - 1]
      const currentLayer = this.parameterizedLayers[i]

      // z(𝓁) = w(𝓁) * a(𝓁-1) + b(𝓁)
      currentLayer.z = fromMatrix(
        addMatrices(
          multiplyMatrices(currentLayer.w, toMatrix(previousLayer.a)),
          toMatrix(currentLayer.b),
        ),
      )

      switch (currentLayer.σ) {
        case 'sigmoid':
          currentLayer.a = currentLayer.z.map(sigmoid)
          break
        case 'relu':
          currentLayer.a = currentLayer.z.map(relu)
          break
        case 'tanh':
          currentLayer.a = currentLayer.z.map(tanh)
          break
        case 'softmax':
          currentLayer.a = softmax(currentLayer.z)
          break
        default: {
          const exhaustiveCheck: never = currentLayer.σ
          throw exhaustiveCheck
        }
      }
    }
  }

  // C = ∑(y - a(𝐋))²
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }

  // ∂C/∂w(𝓁)(𝒿, 𝚔), ∂C/∂b(𝓁)(𝒿), 1 ≤ 𝓁 ≤ 𝐋, 𝒿: layer 𝓁, 𝚔: layer 𝓁-1
  // [
  //   ∂C/∂w(1)(0, 0), ∂C/∂w(1)(0, 1), ∂C/∂w(1)(0, 2), ..., ∂C/∂b(1)(0), ∂C/∂b(1)(1), ... // Layer 2's weights and biases partial derivatives
  //   ...
  //   ∂C/∂w(𝐋-1)(0, 0), ∂C/∂w(𝐋-1)(0, 1), ∂C/∂w(𝐋-1)(0, 2), ..., ∂C/∂b(𝐋-1)(0), ∂C/∂b(𝐋-1)(1), ... // Layer 𝐋-1's weights and biases partial derivatives
  //   ∂C/∂w(𝐋)(0, 0), ∂C/∂w(𝐋)(0, 1), ∂C/∂w(𝐋)(0, 2), ..., ∂C/∂b(𝐋)(0), ∂C/∂b(𝐋)(1), ... // Layer 𝐋's weights and biases partial derivatives
  // ]
  calculateLayerGradient(𝓁: number) {
    const gradient: NumericVector = []

    const previousLayer =
      𝓁 === 1 ? this.inputLayer : this.parameterizedLayers[𝓁 - 2]
    const currentLayer = this.parameterizedLayers[𝓁 - 1]

    timesForEachN(
      [currentLayer.size, previousLayer.size], // Equivalent to: [currentLayer.w.length, currentLayer.w[0].length]
      (row, col) => {
        gradient[row * previousLayer.size + col] =
          previousLayer.a[col] * // ∂z/∂w
          this.activationFnDeriv(currentLayer)(currentLayer.z[row]) * // ∂a/∂z
          (2 * (currentLayer.a[row] - this.y[row])) // ∂C/∂a
      },
    )

    timesForEach(currentLayer.size, i => {
      gradient[currentLayer.size * previousLayer.size + i] =
        1 * // ∂z/∂b
        this.activationFnDeriv(currentLayer)(currentLayer.z[i]) * // ∂a/∂z
        (2 * (currentLayer.a[i] - this.y[i])) // ∂C/∂a
    })

    return gradient
  }

  backPropagate() {}

  private activationFnDeriv(layer: ParameterizedLayer) {
    switch (layer.σ) {
      case 'sigmoid':
        return sigmoidDeriv
      case 'relu':
        return reluDeriv
      case 'tanh':
        return tanhDeriv
      case 'softmax':
        return () => 1
      default: {
        const exhaustiveCheck: never = layer.σ
        throw exhaustiveCheck
      }
    }
  }
}
