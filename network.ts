import { timesMap } from './utils.ts'
import { NumericVector } from './types.ts'
import { MNIST_OUTPUT_SIZE } from './constants.ts'
import { Layer } from './layer.ts'

export class Network {
  y: NumericVector = [] // Expected values for the current sample

  constructor(
    public layers: Layer[],
    public η: number, // Learning rate (greek letter eta)
  ) {
    if (layers[layers.length - 1].size !== MNIST_OUTPUT_SIZE)
      throw new Error(
        `Expected output layer size of ${MNIST_OUTPUT_SIZE} but got ${layers[layers.length - 1].size}`,
      )

    // Fill weights and biases with random data, except for the input layer.
    for (let i = 1; i < this.layers.length; i++)
      this.layers[i].initializeParameters(this.layers[i - 1])
  }

  get inputLayer() {
    return this.layers[0]
  }

  get outputLayer() {
    return this.layers[this.layers.length - 1]
  }

  get parameterCount() {
    return this.layers.reduce((count, layer) => count + layer.parameterCount, 0)
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
    for (let i = 1; i < this.layers.length; i++)
      this.layers[i].calculateActivationValues(this.layers[i - 1])
  }

  // C = ∑(y - a(𝐋))²
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }

  calculateLayerGradient(𝓁: number) {
    return this.layers[𝓁].calculateGradient(this.layers[𝓁 - 1], this.y)
  }

  backPropagate() {}
}
