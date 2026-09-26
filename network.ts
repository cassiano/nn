import {
  timesMap,
  transposeMatrix,
  multiplyMatrices,
  toMatrix,
  fromMatrix,
  addMatrices,
  multiplyMatrixByScalar,
  addVectors,
  multiplyVectorByScalar,
} from './utils.ts'
import {
  NumericVector,
  InitialLayerData,
  LayerGradient,
  Gradient,
} from './types.ts'
import { Layer } from './layer.ts'
import { hadamardProduct } from './utils.ts'
import {
  MNIST_OUTPUT_SIZE,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
} from './mnist_loader.ts'

/**
 * A feedforward neural network made of an ordered list of {@link Layer}s.
 *
 * Owns the training state for the sample currently being processed: the
 * expected output `y` (one-hot label) and the network's cost (MSE). It drives
 * the forward pass by asking each layer beyond the input one to recompute its
 * activations. Backpropagation (weight optimization) is not implemented yet.
 */
export class Network {
  layers: Layer[] = []
  y: NumericVector = [] // Expected values for the current sample (one-hot label vector)

  constructor(
    layersData: InitialLayerData[],
    public η: number, // Learning rate (greek letter eta)
  ) {
    // Build each layer sequentially; only the input layer skips weight
    // initialization because it has no incoming connections.
    for (const data of layersData) {
      const layer = new Layer(this, data.name, data.size, data.σ)

      this.layers.push(layer)
    }

    // The architecture must match the dataset: 28x28 pixels in, 10 digits out.
    if (this.inputLayer.size !== MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
      throw new Error(
        `Expected input layer size of ${MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS} but got ${this.inputLayer.size}`,
      )

    if (this.outputLayer.size !== MNIST_OUTPUT_SIZE)
      throw new Error(
        `Expected output layer size of ${MNIST_OUTPUT_SIZE} but got ${this.outputLayer.size}`,
      )
  }

  /** First layer, which holds the raw inputs and has no weights/bias. */
  get inputLayer() {
    return this.layers[0]
  }

  /** Last layer, whose activations are compared against the expected label. */
  get outputLayer() {
    return this.layers[this.𝐋]
  }

  /** Total number of trainable parameters (weights + biases, input layer excluded). */
  get parameterCount() {
    return this.layers.reduce((count, layer) => count + layer.parameterCount, 0)
  }

  previousLayer(𝓁: number) {
    if (𝓁 === 0) throw new Error('Input layer does not have a previous one')

    return this.layers[𝓁 - 1]
  }

  nextLayer(𝓁: number) {
    if (𝓁 === this.𝐋) throw new Error('Output layer does not have a next one')

    return this.layers[𝓁 + 1]
  }

  /**
   * Registers one training sample: stores the raw pixel values on the input
   * layer and converts the numeric label (0-9) into a one-hot target vector,
   * e.g. label 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
   */
  loadSample(inputs: number[], label: number) {
    if (inputs.length !== this.inputLayer.size)
      throw new Error(
        `Expected input size of ${this.inputLayer.size} but got ${inputs.length}`,
      )

    this.inputLayer.a = inputs

    this.y = timesMap(MNIST_OUTPUT_SIZE, i => (i === label ? 1 : 0))
  }

  /**
   * Forward pass: propagate activations from the input layer to the output
   * layer by recomputing, for every hidden/output layer 𝓁, z(𝓁) and a(𝓁).
   */
  feedForward() {
    for (let 𝓁 = 1; 𝓁 < this.𝐋; 𝓁++)
      this.layers[𝓁].calculatePostActivationValues()
  }

  // Mean squared error between desired vector y and current output a(𝐋):
  // C = (1/n) Σ(yᵢ - a(𝐋)ᵢ)²   (n = number of neurons, 10 here)
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }

  get 𝐋() {
    return this.layers.length - 1
  }

  // Calculates the gradient vector, traversing layers backwards and accumulating weight updates
  // using the chain rule so the network can learn from its errors.
  //
  // ∂C/∂w(𝓁)(𝒿, 𝚔), ∂C/∂b(𝓁)(𝒿), 1 ≤ 𝓁 ≤ 𝐋, 𝒿: layer 𝓁, 𝚔: layer 𝓁-1
  // [
  //   ∂C/∂w(1)(0, 0), ∂C/∂w(1)(0, 1), ∂C/∂w(1)(0, 2), ..., ∂C/∂b(1)(0), ∂C/∂b(1)(1), ... // Layer 2's weights and biases partial derivatives
  //   ...
  //   ∂C/∂w(𝐋-1)(0, 0), ∂C/∂w(𝐋-1)(0, 1), ∂C/∂w(𝐋-1)(0, 2), ..., ∂C/∂b(𝐋-1)(0), ∂C/∂b(𝐋-1)(1), ... // Layer 𝐋-1's weights and biases partial derivatives
  //   ∂C/∂w(𝐋)(0, 0), ∂C/∂w(𝐋)(0, 1), ∂C/∂w(𝐋)(0, 2), ..., ∂C/∂b(𝐋)(0), ∂C/∂b(𝐋)(1), ... // Layer 𝐋's weights and biases partial derivatives
  // ]
  calculateGradient() {
    const gradient: Gradient = []

    for (let 𝓁 = this.𝐋; 𝓁 >= 1; 𝓁--) {
      const currentLayer = this.layers[𝓁]
      const previousLayer = this.previousLayer(𝓁)
      const σDerivatives: NumericVector = currentLayer.z.map(
        currentLayer.σDerivativeFn,
      )

      // [/doc_img/network.ts/2026-09-26-18-25-14.png]
      currentLayer.δ = hadamardProduct(
        𝓁 === this.𝐋
          ? currentLayer.a.map(
              (activationValue, i) => 2 * (activationValue - this.y[i]),
            )
          : fromMatrix(
              multiplyMatrices(
                transposeMatrix(this.nextLayer(𝓁).w),
                toMatrix(this.nextLayer(𝓁).δ),
              ),
            ),
        σDerivatives,
      )

      const layerGradient: LayerGradient = {
        𝓁,
        // [/doc_img/network.ts/2026-09-26-18-21-20.png]
        w: multiplyMatrices(
          toMatrix(currentLayer.δ),
          transposeMatrix(toMatrix(previousLayer.a)),
        ),
        // [/doc_img/network.ts/2026-09-26-18-21-43.png]
        b: currentLayer.δ,
      }

      gradient.push(layerGradient)
    }

    return gradient
  }

  backPropagate() {
    const gradient = this.calculateGradient()

    for (const { 𝓁, b, w } of gradient) {
      const layer = this.layers[𝓁]

      layer.w = addMatrices(layer.w, multiplyMatrixByScalar(w, -this.η))
      layer.b = addVectors(layer.b, multiplyVectorByScalar(b, -this.η))
    }
  }
}
