import {
  timesMap,
  transposeMatrix,
  addMatrices,
  multiplyMatrixByScalar,
  addVectors,
  multiplyVectorByScalar,
  createMatrix,
  multiplyMatrixByVector,
  outerProduct,
} from './utils.ts'
import {
  NumericVector,
  InitialLayerData,
  GradientLayer,
  Gradient,
  NetworkOptions,
} from './types.ts'
import { Layer } from './layer.ts'
import { hadamardProduct, createVector, divideMatrixByScalar } from './utils.ts'
import { divideVectorByScalar, timesForEach } from './utils.ts'

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
   * How many samples each weight update averages over. The gradient of a
   * batch is a mean, so the update scales by this to keep η a per-sample rate.
   */
  batchSize: number

  /**
   * Builds a network from a high-level description of each layer.
   *
   * The network knows nothing about any particular dataset: the expected input
   * and output sizes are optional checks, so the same class serves MNIST
   * (784 in, 10 out) and XOR (2 in, 2 out).
   *
   * @param layersData One entry per layer, in feedforward order.
   * @param η Learning rate, used by {@link Network.backPropagate} to scale each
   * update.
   * @param options Optional expectations about the data and the batch size.
   * @returns Nothing; the layers are stored on the network.
   * @throws If the architecture does not match the expected input/output sizes,
   * or if the batch size is not positive.
   */
  constructor(
    layersData: InitialLayerData[],
    public η: number, // Learning rate (greek letter eta)
    options: NetworkOptions = {},
  ) {
    for (const data of layersData) {
      const layer = new Layer(this, data.name, data.size, data.σ)

      this.layers.push(layer)
    }

    this.batchSize = options.batchSize ?? 1

    if (this.batchSize < 1)
      throw new Error(
        `Expected batch size of at least 1 but got ${this.batchSize}`,
      )

    const { inputSize, outputSize } = options

    if (inputSize !== undefined && this.inputLayer.size !== inputSize)
      throw new Error(
        `Expected input layer size of ${inputSize} but got ${this.inputLayer.size}`,
      )

    if (outputSize !== undefined && this.outputLayer.size !== outputSize)
      throw new Error(
        `Expected output layer size of ${outputSize} but got ${this.outputLayer.size}`,
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
   * label as a one-hot target in {@link Network.y}. Does not run a forward pass.
   *
   * The target has one entry per output neuron, so a 10-class network encodes
   * `3` as [0,0,0,1,0,0,0,0,0,0] and a 2-class one encodes `1` as [0,1].
   *
   * @param inputs One value per input neuron.
   * @param label The class the sample belongs to, 0-based.
   * @returns Nothing; the sample is stored on the network.
   * @throws If `inputs` is not the input layer's size, or if `label` is not a
   * class the output layer can hold.
   */
  loadSample(inputs: NumericVector, label: number): void {
    if (inputs.length !== this.inputLayer.size)
      throw new Error(
        `Expected input size of ${this.inputLayer.size} but got ${inputs.length}`,
      )

    if (!Number.isInteger(label) || label < 0 || label >= this.outputLayer.size)
      throw new Error(
        `Expected label between 0 and ${this.outputLayer.size - 1} but got ${label}`,
      )

    this.inputLayer.a = inputs

    this.y = timesMap(this.outputLayer.size, i => (i === label ? 1 : 0))
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
   * The class the network currently predicts, given a prior forward pass: the
   * output neuron with the highest activation.
   *
   * @returns The index of the highest activation, 0-based.
   */
  predictedClass(): number {
    const highestActivation = Math.max(...this.outputLayer.a)

    return this.outputLayer.a.findIndex(value => value === highestActivation)
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

    // [/doc_img/network.ts/2026-10-19-09-55-27.png]
    // [/doc_img/network.ts/2026-10-19-09-55-43.png]
    // [/doc_img/network.ts/2026-10-21-09-46-41.png]
    for (let 𝓁 = this.𝐋; 𝓁 >= 1; 𝓁--) {
      const layer = this.layers[𝓁]
      const σDerivatives: NumericVector = layer.z.map(layer.σDerivativeFn)

      let costByActivationPartialDerivatives: NumericVector // ∂C/∂a

      if (layer.isOutputLayer) {
        costByActivationPartialDerivatives = layer.a.map(
          (activationValue, i) => 2 * (activationValue - this.y[i]),
        )
      } else {
        const nextLayer = layer.nextLayer

        costByActivationPartialDerivatives = multiplyMatrixByVector(
          transposeMatrix(nextLayer.w),
          nextLayer.δ,
        )
      }

      // [/doc_img/network.ts/2026-09-26-18-25-14.png]
      layer.δ = hadamardProduct(
        costByActivationPartialDerivatives,
        σDerivatives,
      )

      const gradientlayer: GradientLayer = {
        𝓁,
        w: outerProduct(layer.δ, layer.previousLayer.a), // [/doc_img/network.ts/2026-09-26-18-21-20.png]
        b: [...layer.δ], // [/doc_img/network.ts/2026-09-26-18-21-43.png]
      }

      gradient.push(gradientlayer)
    }

    return gradient
  }

  /**
   * One gradient-descent step over every trainable parameter:
   *
   *   w(𝓁) ← w(𝓁) - η·batchSize·∂C/∂w(𝓁)
   *   b(𝓁) ← b(𝓁) - η·batchSize·∂C/∂b(𝓁)
   *
   * batchSize cancels the averaging done in
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

      if (layer === undefined) throw new Error(`Layer ${𝓁} not found`)

      layer.w = addMatrices(
        layer.w,
        multiplyMatrixByScalar(w, -this.η * this.batchSize),
      )

      layer.b = addVectors(
        layer.b,
        multiplyVectorByScalar(b, -this.η * this.batchSize),
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

      // Accumulate in place: a fresh matrix/vector per sample would allocate
      // hundreds of thousands of cells per batch for the 784-wide first layer.
      const summedW = createMatrix(w.length, w[0].length, 0)
      const summedB = createVector(b.length, 0)

      for (const gradient of gradients) {
        const layer = gradient[layerIdx]

        if (layer.𝓁 !== 𝓁)
          throw new Error(`Mixing distinct 𝓁 values (${layer.𝓁} and ${𝓁})`)

        for (let row = 0; row < summedW.length; row++) {
          const summedRow = summedW[row]
          const layerRow = layer.w[row]

          for (let col = 0; col < summedRow.length; col++)
            summedRow[col] += layerRow[col]
        }

        for (let i = 0; i < summedB.length; i++) summedB[i] += layer.b[i]
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

  async saveToDisk(path: string) {
    // Save network weights to disk for later inspection or reuse.
    await Deno.writeTextFile(
      path,
      JSON.stringify(
        this.layers
          .slice(1)
          .map(layer => ({ 𝓁: layer.𝓁, w: layer.w, b: layer.b })),
        null,
        2,
      ),
    )
  }
}
