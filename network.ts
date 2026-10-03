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
  createMatrix,
} from './utils.ts'
import {
  NumericVector,
  InitialLayerData,
  GradientLayer,
  Gradient,
} from './types.ts'
import { Layer } from './layer.ts'
import {
  hadamardProduct,
  createVector,
  divideMatrixByScalar,
  assertIsNotUndefined,
} from './utils.ts'
import { BATCH_SIZE } from './constants.ts'
import { divideVectorByScalar, timesForEach } from './utils.ts'
import {
  MNIST_OUTPUT_SIZE,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
} from './mnist_loader.ts'

/**
 * A feedforward network made of an ordered list of {@link Layer}s, indexed by
 * 𝓁: 0 is the parameterless input layer, 𝐋 the output layer.
 *
 * It holds the state of the sample being processed (its target y and the cost)
 * and drives both passes: forward through the layers, then gradients stepped
 * against the parameters. Since only one sample is loaded at a time, a batch is
 * trained by collecting one gradient per sample, averaging them with
 * {@link Network.calculateAverageGradient} and applying that single step.
 */
export class Network {
  /** The layers, input first and output last. */
  layers: Layer[] = []

  /** The current sample's target as a one-hot vector, empty until one is loaded. */
  y: NumericVector = []

  /**
   * Builds a network from a high-level description of each layer.
   *
   * @param layersData One entry per layer, in feedforward order.
   * @param η Learning rate, used by {@link Network.backPropagate} to scale each
   * update.
   * @returns Nothing; the layers are stored on the network.
   * @throws If the architecture does not match the dataset (784 in, 10 out).
   */
  constructor(
    layersData: InitialLayerData[],
    public η: number, // Learning rate (greek letter eta)
  ) {
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

  /** First layer, holding the raw inputs. */
  get inputLayer(): Layer {
    return this.layers[0]
  }

  /** Last layer, whose activations are the network's prediction. */
  get outputLayer(): Layer {
    return this.layers[this.𝐋]
  }

  /** Total number of trainable parameters across the layers. */
  get parameterCount(): number {
    return this.layers.reduce((count, layer) => count + layer.parameterCount, 0)
  }

  /**
   * The layer feeding into layer 𝓁.
   *
   * @param 𝓁 Index of the layer whose predecessor is wanted.
   * @returns The layer at 𝓁 - 1.
   * @throws If 𝓁 is 0, the input layer having no predecessor.
   */
  previousLayer(𝓁: number): Layer {
    if (𝓁 === 0) throw new Error('Input layer does not have a previous one')

    return this.layers[𝓁 - 1]
  }

  /**
   * The layer fed by layer 𝓁.
   *
   * @param 𝓁 Index of the layer whose successor is wanted.
   * @returns The layer at 𝓁 + 1.
   * @throws If 𝓁 is 𝐋, the output layer having no successor.
   */
  nextLayer(𝓁: number): Layer {
    if (𝓁 === this.𝐋) throw new Error('Output layer does not have a next one')

    return this.layers[𝓁 + 1]
  }

  /**
   * Registers a training sample: the raw inputs on the input layer, and the
   * digit as a one-hot target in {@link Network.y}. Does not run a forward pass.
   *
   * @param inputs One value per input neuron, expected normalized to [0, 1].
   * @param label The digit the image depicts, 0-9.
   * @returns Nothing; the sample is stored on the network.
   * @throws If `inputs` is not the input layer's size.
   */
  loadSample(inputs: NumericVector, label: number): void {
    if (inputs.length !== this.inputLayer.size)
      throw new Error(
        `Expected input size of ${this.inputLayer.size} but got ${inputs.length}`,
      )

    this.inputLayer.a = inputs

    this.y = timesMap(MNIST_OUTPUT_SIZE, i => (i === label ? 1 : 0))
  }

  /**
   * Forward pass: recompute a and z from the input layer up to the output layer.
   *
   * @returns Nothing; each layer's state is updated in place.
   */
  feedForward(): void {
    for (let 𝓁 = 1; 𝓁 <= this.𝐋; 𝓁++)
      this.layers[𝓁].calculatePostActivationValues()
  }

  /**
   * Squared error between the target and the output layer's activations:
   * C = Σ(yᵢ - a(𝐋)ᵢ)².
   *
   * Unnormalized, which only rescales the gradient: the batch mean divides by
   * the batch size and the update multiplies it back, so η stays the per-sample
   * rate.
   */
  get cost(): number {
    return timesMap(
      this.outputLayer.a.length,
      i => (this.y[i] - this.outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }

  /** Index of the output layer, which is also how many layers are trainable. */
  get 𝐋(): number {
    return this.layers.length - 1
  }

  /**
   * The digit the network currently predicts, given a prior forward pass.
   *
   * @returns The index of the highest activation, 0-9.
   */
  predictedDigit(): number {
    const predictedDigitProbability = Math.max(...this.outputLayer.a)

    return this.outputLayer.a.findIndex(
      value => value === predictedDigitProbability,
    )
  }

  /**
   * The gradient of {@link Network.cost} with respect to every trainable
   * parameter, by the chain rule, from 𝐋 back to 1. Per layer:
   *
   *   δ(𝓁) = ∂C/∂a(𝓁) ☉ σ'(z(𝓁))     ∂C/∂w(𝓁) = δ(𝓁)·a(𝓁-1)ᵀ
   *   ∂C/∂a(𝐋) = 2·(a(𝐋) - y)          ∂C/∂b(𝓁) = δ(𝓁)
   *
   * Inner layers take ∂C/∂a(𝓁) = w(𝓁+1)ᵀ·δ(𝓁+1) from the layer already visited,
   * which is why the traversal runs backwards. Needs a loaded sample and a
   * prior {@link Network.feedForward}.
   *
   * @returns One {@link GradientLayer} per trainable layer, output layer first.
   * @throws If a layer is missing its activation function, needed for σ'.
   */
  calculateGradient(): Gradient {
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
      let costByActivationDerivatives: NumericVector // ∂C/∂a

      if (currentLayer.isOutputLayer) {
        costByActivationDerivatives = currentLayer.a.map(
          (activationValue, i) => 2 * (activationValue - this.y[i]),
        )
      } else {
        assertIsNotUndefined(nextLayer)

        costByActivationDerivatives = fromMatrix(
          multiplyMatrices(transposeMatrix(nextLayer.w), toMatrix(nextLayer.δ)),
        )
      }

      // [/doc_img/network.ts/2026-09-26-18-25-14.png]
      currentLayer.δ = hadamardProduct(
        costByActivationDerivatives,
        σDerivatives,
      )

      const gradientlayer: GradientLayer = {
        𝓁,
        // [/doc_img/network.ts/2026-09-26-18-21-20.png]
        w: multiplyMatrices(
          toMatrix(currentLayer.δ),
          transposeMatrix(toMatrix(previousLayer.a)),
        ),
        // [/doc_img/network.ts/2026-09-26-18-21-43.png]
        b: [...currentLayer.δ],
      }

      // Transposing the factors instead is equivalent; see the test in
      // `tests/utils_test.ts` for the proof that both products match.
      gradient.push(gradientlayer)
    }

    return gradient
  }

  /**
   * One gradient-descent step over every trainable parameter:
   *
   *   w(𝓁) ← w(𝓁) - η·BATCH_SIZE·∂C/∂w(𝓁)
   *   b(𝓁) ← b(𝓁) - η·BATCH_SIZE·∂C/∂b(𝓁)
   *
   * BATCH_SIZE cancels the averaging done in
   * {@link Network.calculateAverageGradient}, so η remains the per-sample rate
   * and a batch of a different length is not compensated for.
   *
   * Entries are matched to layers by their own 𝓁, so a per-sample gradient
   * gives an SGD step and a batch mean gives a mini-batch step.
   *
   * @param gradient The gradients to step against, as returned by
   * {@link Network.calculateGradient} or its average.
   * @returns Nothing; each layer's parameters are updated in place.
   * @throws If an entry's 𝓁 is not a layer of this network, or if a gradient
   * shape does not line up with the parameters it is applied to.
   */
  backPropagate(gradient: Gradient): void {
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

  /**
   * Averages a batch of per-sample gradients into one, so a whole batch can be
   * applied in a single {@link Network.backPropagate} step.
   *
   * Layers are paired by array position, and every sample's entry at a position
   * must carry the same 𝓁 as the first one's, so the batch has to agree on the
   * order {@link Network.calculateGradient} produced — output layer first, not
   * ascending.
   *
   * @param gradients One {@link Gradient} per sample.
   * @returns The mean gradient, in the first sample's order.
   * @throws If `gradients` is empty, or if the samples disagree about the layers.
   */
  calculateAverageGradient(gradients: Gradient[]): Gradient {
    if (gradients.length === 0)
      throw new Error(`Cannot calculate average gradient (empty collection)`)

    const firstGradient = gradients[0]
    const averageGradient: Gradient = []

    // Average w and b per layer, pairing entry layerIdx of every sample; the
    // first sample supplies the 𝓁 labels and the entry order.
    timesForEach(this.𝐋, layerIdx => {
      const { w, b, 𝓁 } = firstGradient[layerIdx]

      let summedW = createMatrix(w.length, w[0].length, 0)
      let summedB = createVector(b.length, 0)

      for (const gradient of gradients) {
        const layer = gradient[layerIdx]

        if (layer.𝓁 !== 𝓁)
          throw new Error(`Mixing distinct 𝓁 values (${layer.𝓁} and ${𝓁})`)

        summedW = addMatrices(summedW, layer.w)
        summedB = addVectors(summedB, layer.b)
      }

      const averageGradientLayer: GradientLayer = {
        𝓁,
        w: divideMatrixByScalar(summedW, gradients.length),
        b: divideVectorByScalar(summedB, gradients.length),
      }

      averageGradient.push(averageGradientLayer)
    })

    return averageGradient
  }
}
