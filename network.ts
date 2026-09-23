import {
  random,
  timesMap,
  assertIsNotUndefined,
  fromMatrix,
  addMatrices,
  multiplyMatrices,
  toMatrix,
  timesMapN,
} from './utils.ts'
import { sigmoid, relu, tanh } from './activation.ts'
import { softmax } from './activation.ts'
import { Layer, NumericVector } from './types.ts'
import { MNIST_OUTPUT_SIZE } from './constants.ts'

export class Network {
  y: NumericVector = [] // Expected values

  constructor(public layers: Layer[]) {
    if (this.layers[this.layers.length - 1].size !== MNIST_OUTPUT_SIZE)
      throw new Error(
        `Expected output layer size of ${MNIST_OUTPUT_SIZE} but got ${this.layers[this.layers.length - 1].size}`,
      )

    // Fill weights and biases of all but the input layer with random data.
    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].w = timesMapN(
        [this.layers[i].size, layers[i - 1].size],
        () => random(-1, 1),
      )

      this.layers[i].b = timesMap(this.layers[i].size, () => random(-1, 1))
    }
  }

  get parameterCount() {
    return this.layers
      .slice(1)
      .reduce(
        (count, layer) =>
          count + layer.w!.length * layer.w![0].length + layer.b!.length,
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
    for (let i = 1; i < this.layers.length; i++) {
      // console.log(`Processing layer ${i}: ${this.layers[i].name}`)

      const currentLayer = this.layers[i]
      const previousLayer = this.layers[i - 1]

      assertIsNotUndefined(currentLayer.σ)
      assertIsNotUndefined(currentLayer.w)
      assertIsNotUndefined(currentLayer.b)

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

  get inputLayer() {
    return this.layers[0]
  }

  get outputLayer() {
    return this.layers[this.layers.length - 1]
  }

  // C = ∑(y - a(𝐋))²
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }
}
