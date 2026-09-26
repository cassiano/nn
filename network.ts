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
 * expected output `y` (one-hot label) and the network's cost. It drives the
 * forward pass by asking each layer beyond the input one to recompute its
 * activations, and the backward pass by computing per-layer gradients and
 * stepping the weights against them.
 *
 * Layers are identified by index 𝓁: 0 is the input layer, which has no
 * parameters, and {@link Network.𝐋} is the output layer.
 */
export class Network {
  /**
   * The ordered list of layers, input first and output last. Built by the
   * constructor; index 𝓁 matches each layer's own `𝓁`.
   */
  layers: Layer[] = []

  /**
   * Expected values for the current sample as a one-hot vector, set by
   * {@link Network.loadSample}. Stays empty until a sample is loaded.
   */
  y: NumericVector = []

  /**
   * Builds a network from a high-level description of each layer.
   *
   * @param layersData One entry per layer, in feedforward order. The first
   * entry must be the input layer and the last the output layer; their sizes
   * are validated against the MNIST dimensions (784 in, 10 out).
   * @param η Learning rate (greek letter eta), stored as a public field and
   * reserved for the weight updates that {@link Network.backPropagate} will
   * perform. Currently unused.
   * @throws If the first layer's size is not 784, or the last layer's size is
   * not 10.
   */
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

  /**
   * Returns the layer feeding into layer 𝓁, i.e. the one whose activations act
   * as that layer's input.
   *
   * @param 𝓁 Index of the layer whose predecessor is wanted, where 0 is the
   * input layer and 1 is the first trainable layer.
   * @returns The layer at index `𝓁 - 1`.
   * @throws If `𝓁` is 0, since the input layer has no predecessor.
   */
  previousLayer(𝓁: number) {
    if (𝓁 === 0) throw new Error('Input layer does not have a previous one')

    return this.layers[𝓁 - 1]
  }

  /**
   * Returns the layer fed by layer 𝓁, used during backpropagation to carry the
   * error signal one step further back.
   *
   * @param 𝓁 Index of the layer whose successor is wanted, where 0 is the
   * input layer and 𝐋 is the output layer.
   * @returns The layer at index `𝓁 + 1`.
   * @throws If `𝓁` equals {@link Network.𝐋}, since the output layer has no
   * successor.
   */
  nextLayer(𝓁: number) {
    if (𝓁 === this.𝐋) throw new Error('Output layer does not have a next one')

    return this.layers[𝓁 + 1]
  }

  /**
   * Registers one training sample: stores the raw pixel values on the input
   * layer and converts the numeric label (0-9) into a one-hot target vector,
   * e.g. label 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
   *
   * Does not run a forward pass; call {@link Network.feedForward} afterwards to
   * propagate the sample through the layers.
   *
   * @param inputs Raw pixel values for the sample, one per input neuron,
   * expected to be normalized to [0, 1] and of length equal to
   * {@link Network.inputLayer}'s size (784 for MNIST).
   * @param label Digit the image depicts, 0-9. Exactly one entry of the
   * resulting {@link Network.y} becomes 1; the rest become 0.
   * @throws If `inputs.length` differs from the input layer's size.
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
   *
   * Reads {@link Network.inputLayer}'s `a`, so {@link Network.loadSample} (or a
   * manual assignment) must run first. Leaves δ from any previous
   * {@link Network.calculateGradient} untouched; only backpropagation writes
   * it.
   */
  feedForward() {
    for (let 𝓁 = 1; 𝓁 <= this.𝐋; 𝓁++)
      this.layers[𝓁].calculatePostActivationValues()
  }

  /**
   * Mean squared error between the target vector y and the current output
   * a(𝐋): C = Σ(yᵢ - a(𝐋)ᵢ)², summed over the output neurons.
   *
   * Note there is no 1/n normalization here, which only rescales the gradient
   * and therefore the effective learning rate.
   */
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }

  /**
   * Index of the last (output) layer, i.e. the total number of layers minus
   * one. Convenient shorthand for walking the network's tail.
   */
  get 𝐋() {
    return this.layers.length - 1
  }

  /**
   * Computes the gradient of {@link Network.cost} with respect to every
   * trainable parameter, by the chain rule, traversing layers backwards.
   *
   * Works in two steps per layer 𝓁, from 𝐋 down to 1:
   * 1. Error signal δ(𝓁) = ∂C/∂a(𝓁) ☉ σ'(z(𝓁)). For the output layer (𝓁 = 𝐋)
   *    ∂C/∂a(𝐋)ᵢ = 2·(a(𝐋)ᵢ - yᵢ) comes straight from the squared cost. For an
   *    inner layer it is propagated from the next one:
   *    ∂C/∂a(𝓁) = w(𝓁+1)ᵀ · δ(𝓁+1), which is why the traversal must run
   *    backwards and why `nextLayer(𝓁).δ` is already available.
   * 2. Parameter partials: ∂C/∂w(𝓁)(𝒿, 𝚔) = δ(𝓁)𝒿 · a(𝓁-1)𝚔 and
   *    ∂C/∂b(𝓁)𝒿 = δ(𝓁)𝒿.
   *
   * Requires {@link Network.y} and a prior {@link Network.feedForward}, since
   * δ is derived from the sample's activations and pre-activations.
   *
   * @returns A {@link Gradient}: one {@link LayerGradient} per trainable layer,
   * ordered output layer (𝓁 = 𝐋) first, then each inner layer descending to 1.
   * Each `w` is shaped [size][previous layer size] and each `b` is [size], so
   * they can be applied straight onto the matching parameters.
   * @throws If any layer except the input has no activation function σ
   * configured, since σ' is needed to build δ.
   */
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
        currentLayer.isOutputLayer
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

  /**
   * Performs one gradient-descent step over all trainable parameters, using
   * the learning rate {@link Network.η}:
   *
   *   w(𝓁) ← w(𝓁) - η·∂C/∂w(𝓁)
   *   b(𝓁) ← b(𝓁) - η·∂C/∂b(𝓁)
   *
   * Because ∂C is added with a negated, scaled matrix/vector, weights and biases
   * move *against* the gradient, reducing the cost on the current sample. Calls
   * {@link Network.calculateGradient}, so {@link Network.feedForward} and
   * {@link Network.loadSample} must have run first.
   *
   * Mutates each layer's `w` and `b` in place (replacing them with new arrays)
   * and returns nothing; a single sample per call, so this is the
   * stochastic-gradient-descent step and an epoch is a loop over samples.
   *
   * @throws If any layer except the input has no activation function σ
   * configured, or if a weight/bias shape does not line up with its gradient.
   */
  backPropagate() {
    const gradient = this.calculateGradient()

    for (const { 𝓁, b, w } of gradient) {
      const layer = this.layers[𝓁]

      layer.w = addMatrices(layer.w, multiplyMatrixByScalar(w, -this.η))
      layer.b = addVectors(layer.b, multiplyVectorByScalar(b, -this.η))
    }
  }

  predictedDigit() {
    const predictedDigitPercentage = Math.max(...this.outputLayer.a)

    return this.outputLayer.a.findIndex(
      value => value === predictedDigitPercentage,
    )
  }
}
