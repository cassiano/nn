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
import { timesForEachN } from './utils.ts'

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
}
