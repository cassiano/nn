import { timesMap } from './utils.ts'
import { NumericVector, InitialLayerData } from './types.ts'
import { Layer } from './layer.ts'
import {
  MNIST_OUTPUT_SIZE,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
} from './mnist_loader.ts'

export class Network {
  layers: Layer[] = []
  y: NumericVector = [] // Expected values for the current sample

  constructor(
    layersData: InitialLayerData[],
    public η: number, // Learning rate (greek letter eta)
  ) {
    // Fill all layers' weights and biases with random data, except for the input layer.
    for (const data of layersData) {
      const layer = new Layer(this, data.name, data.size, data.σ ?? 'none')

      this.layers.push(layer)
    }

    if (this.inputLayer.size !== MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
      throw new Error(
        `Expected input layer size of ${MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS} but got ${this.inputLayer.size}`,
      )

    if (this.outputLayer.size !== MNIST_OUTPUT_SIZE)
      throw new Error(
        `Expected output layer size of ${MNIST_OUTPUT_SIZE} but got ${this.outputLayer.size}`,
      )
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
    for (let 𝓁 = 1; 𝓁 < this.layers.length; 𝓁++)
      this.layers[𝓁].calculateActivationValues()
  }

  // C = ∑(y - a(𝐋))²
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }

  calculateLayerGradient(𝓁: number) {
    return this.layers[𝓁].calculateGradient()
  }

  backPropagate() {}
}
