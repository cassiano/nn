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
import { BATCH_SIZE } from './constants.ts'
import {
  MNIST_OUTPUT_SIZE,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
} from './mnist_loader.ts'

/**
 * A feedforward neural network made of an ordered list of {@link Layer}s.
 *
 * Owns the training state for the sample currently being processed: the
 * expected output `y` (one-hot label) and the network's cost. Only one sample
 * is loaded at a time, so batching happens outside the network: a batch is
 * trained by looping {@link Network.loadSample} → {@link Network.feedForward} →
 * {@link Network.calculateGradient} per sample, averaging the resulting
 * gradients with {@link Network.calculateAverageGradient} and applying that one
 * mean in a single {@link Network.backPropagate} step.
 *
 * It drives the forward pass by asking each layer beyond the input one to
 * recompute its activations, and the backward pass by computing per-layer
 * gradients and stepping the weights against them.
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
   * used by {@link Network.backPropagate} to scale every gradient it applies.
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
   * manual assignment) must run first. Leaves every layer's δ untouched;
   * {@link Network.calculateGradient} is the only writer of δ.
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
   * and therefore the effective learning rate: averaging a batch of n gradients
   * before {@link Network.backPropagate} makes the effective rate η/n.
   */
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }

  /**
   * Index of the last (output) layer, i.e. the total number of layers minus
   * one. Convenient shorthand for walking the network's tail. Doubles as the
   * count of trainable layers, since 0 is the parameterless input layer, which
   * is why {@link Network.calculateAverageGradient} can size its loop with it.
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
   * they can be applied straight onto the matching parameters. Note each `b` is
   * the layer's own δ array handed over by reference, not a copy, so a caller
   * that mutates it in place also mutates the layer.
   * @throws If any layer except the input has no activation function σ
   * configured, since σ' is needed to build δ.
   */
  calculateGradient() {
    const gradient: Gradient = []

    for (let 𝓁 = this.𝐋; 𝓁 >= 1; 𝓁--) {
      const currentLayer = this.layers[𝓁]
      const previousLayer = this.previousLayer(𝓁)
      const nextLayer = !currentLayer.isOutputLayer
        ? this.nextLayer(𝓁)
        : undefined
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
                transposeMatrix(nextLayer!.w),
                toMatrix(nextLayer!.δ),
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
        b: [...currentLayer.δ],
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
   * move *against* the gradient, reducing the cost on the current sample.
   *
   * The gradient is applied as it arrives, so one call covers whichever samples
   * it already summarizes: a single per-sample {@link Gradient} gives an SGD
   * step, and the batch mean from {@link Network.calculateAverageGradient} gives
   * a mini-batch step. An epoch is a loop over those steps, not over samples.
   *
   * Mutates each layer's `w` and `b` in place (replacing them with new arrays)
   * and returns nothing. Entries are addressed by their own {@link
   * LayerGradient.𝓁} index, so a gradient from a network with a different
   * layer count is applied as-is rather than rejected.
   *
   * @throws If a weight or bias shape does not line up with its gradient (from
   * {@link addMatrices}/{@link addVectors}), or with a TypeError if an entry's
   * `𝓁` is not a valid layer index.
   */
  backPropagate(gradient: Gradient) {
    for (const { 𝓁, b, w } of gradient) {
      const layer = this.layers[𝓁]
      if (!layer) throw new Error(`Layer ${𝓁} not found`)

      layer.w = addMatrices(
        layer.w,
        multiplyMatrixByScalar(w, -this.η * BATCH_SIZE),
      )
      layer.b = addVectors(
        layer.b,
        multiplyVectorByScalar(b, -this.η * BATCH_SIZE),
      )
    }
  }

  predictedDigit() {
    const predictedDigitProbability = Math.max(...this.outputLayer.a)

    return this.outputLayer.a.findIndex(
      value => value === predictedDigitProbability,
    )
  }

  /**
   * Averages a batch of per-sample gradients into one gradient, so a whole
   * batch can be applied in a single {@link Network.backPropagate} step.
   *
   * Every sample contributes weight exactly `1 / gradients.length`, making this
   * the true mean of the batch rather than a weighted or rescaled sum. The
   * running sum starts from a zeroed copy of the first entry's shapes and every
   * sample, the first one included, is added into it exactly once, so no sample
   * is counted twice.
   *
   * Layers are paired up **by array position**: entry `i` of every sample is
   * averaged together. As a guard against a batch whose samples disagree, each
   * sample's entry at position `i` must carry the same
   * {@link LayerGradient.𝓁} as the first sample's, and the result inherits
   * that `𝓁` and the first sample's order. The order therefore has to agree
   * across the batch, but it need not ascend by `𝓁`: whatever
   * {@link Network.calculateGradient} produced works, output layer first.
   * Entries past position `this.𝐋` are ignored.
   *
   * Nothing is mutated: the sum starts from a zeroed copy and the helpers all
   * return fresh arrays, so the batch survives the call — including the δ
   * arrays {@link Network.calculateGradient} shares with the layers.
   *
   * @param gradients One {@link Gradient} per sample, each covering the same
   * trainable layers in the same order.
   * @returns The mean {@link Gradient}, one {@link LayerGradient} per trainable
   * layer in the first sample's order, whose `w` and `b` have the first
   * sample's shapes.
   * @throws If `gradients` is empty, if two samples disagree about the `𝓁` at
   * some position, with a TypeError if a sample has fewer entries than this
   * network has trainable layers, or if a summed `w`/`b` shape disagrees with
   * the running sum (from {@link addMatrices}/{@link addVectors}).
   */
  calculateAverageGradient(gradients: Gradient[]): Gradient {
    if (gradients.length === 0)
      throw new Error(`Cannot calculate average gradient (empty collection)`)

    const firstGradient = gradients[0]
    const size = gradients.length
    const averageGradient: Gradient = []

    // Calculate the w and b averages per layer, pairing entry i of every
    // sample (the first sample supplies the 𝓁 labels and the entry order).
    for (let i = 0; i < this.𝐋; i++) {
      const 𝓁 = firstGradient[i].𝓁
      const { w, b } = firstGradient[i]

      // Initialize both sums with 0-filled matrixes/vectors.
      let summedW = initializeMatrix(w.length, w[0].length, 0)
      let summedB = initializeVector(b.length, 0)

      for (const gradient of gradients) {
        if (gradient[i].𝓁 !== 𝓁)
          throw new Error(
            `Mixing distinct 𝓁 values (${gradient[i].𝓁} and ${𝓁})`,
          )

        summedW = addMatrices(summedW, gradient[i].w)
        summedB = addVectors(summedB, gradient[i].b)
      }

      const averageLayerGradient: LayerGradient = {
        𝓁,
        w: multiplyMatrixByScalar(summedW, 1 / size),
        b: multiplyVectorByScalar(summedB, 1 / size),
      }

      averageGradient.push(averageLayerGradient)
    }

    return averageGradient
  }
}
