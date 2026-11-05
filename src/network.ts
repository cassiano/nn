import {
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
  NetworkLayersInfo,
  NetworkParameters,
  CostFunctionType,
} from './types.ts'
import { Layer } from './layer.ts'
import { hadamardProduct, createVector, divideMatrixByScalar } from './utils.ts'
import { divideVectorByScalar, timesForEach } from './utils.ts'
import { costDerivative, costValue } from './cost.ts'

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

  /** Whether biases are used in the network. */
  useBias: boolean

  /** Cost minimized by training: squared error or cross-entropy. */
  costFunction: CostFunctionType

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
   *
   * @example
   * const network = new Network(
   *   [
   *     { name: 'input', size: 2 },
   *     { name: 'hidden', size: 4, σ: 'sigmoid' },
   *     { name: 'output', size: 2, σ: 'softmax' },
   *   ],
   *   0.5,
   *   { inputSize: 2, outputSize: 2, batchSize: 4 },
   * )
   * network.parameterCount // 22
   */
  constructor(
    layersData: InitialLayerData[],
    public η: number, // Learning rate (greek letter eta)
    options: NetworkOptions = {},
  ) {
    this.useBias = options.useBias ?? true
    this.costFunction = options.costFunction ?? 'mse'

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

  /**
   * First layer, holding the raw inputs.
   *
   * @example
   * network.inputLayer.size // 2
   */
  get inputLayer(): Layer {
    return this.layers[0]
  }

  /**
   * Last layer, whose activations are the network's prediction.
   *
   * @example
   * network.outputLayer.size // 2
   */
  get outputLayer(): Layer {
    return this.layers[this.𝐋]
  }

  /**
   * Total number of trainable parameters across the layers.
   *
   * @example
   * // 2 -> 4 -> 2: (4*2 + 4) + (2*4 + 2)
   * network.parameterCount // 22
   */
  get parameterCount(): number {
    return this.layers.reduce((count, layer) => count + layer.parameterCount, 0)
  }

  /**
   * The layer feeding into layer 𝓁.
   *
   * @param 𝓁 Index of the layer whose predecessor is wanted.
   * @returns The layer at 𝓁 - 1.
   * @throws If 𝓁 is 0, the input layer having no predecessor.
   *
   * @example
   * network.previousLayer(2) === network.layers[1] // true
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
   *
   * @example
   * network.nextLayer(1) === network.layers[2] // true
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
   *
   * @example
   * network.loadSample([0, 1], 1)
   * network.inputLayer.a // [0, 1]
   * network.y // [0, 1]
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

    this.y = Array(this.outputLayer.size).fill(0)
    this.y[label] = 1
  }

  /**
   * Forward pass: recompute a and z from the input layer up to the output layer.
   *
   * @returns Nothing; each layer's state is updated in place.
   *
   * @example
   * network.loadSample([0, 1], 1)
   * network.feedForward()
   * network.outputLayer.a.length // 2
   */
  feedForward(): void {
    for (let 𝓁 = 1; 𝓁 <= this.𝐋; 𝓁++)
      this.layers[𝓁].calculatePostActivationValues()
  }

  /**
   * Cost between the target and the output layer's activations, selected by
   * {@link Network.costFunction}: squared error C = Σ(yᵢ - a(𝐋)ᵢ)² or
   * cross-entropy C = -Σ yᵢ·ln(a(𝐋)ᵢ).
   *
   * Unnormalized, which only rescales the gradient: the batch mean divides by
   * the batch size and the update multiplies it back, so η stays the per-sample
   * rate.
   *
   * @example
   * network.y = [1, 0]
   * network.outputLayer.a = [1, 0]
   * network.cost // 0
   */
  get cost(): number {
    return costValue(this.costFunction, this.y, this.outputLayer.a)
  }

  /**
   * Index of the output layer, which is also how many layers are trainable.
   *
   * @example
   * network.𝐋 // 2 (input at 0, hidden at 1, output at 2)
   */
  get 𝐋(): number {
    return this.layers.length - 1
  }

  /**
   * The class the network currently predicts, given a prior forward pass: the
   * output neuron with the highest activation.
   *
   * @returns The index of the highest activation, 0-based.
   *
   * @example
   * network.outputLayer.a = [0.1, 0.9]
   * network.predictedClass() // 1
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
   *   ∂C/∂a(𝐋) from the selected cost ∂C/∂b(𝓁) = δ(𝓁)
   *
   * Softmax is not element-wise: its Jacobian replaces the ☉ σ' step. Composed
   * with cross-entropy that collapses to δ = a - y. Inner layers take
   * ∂C/∂a(𝓁) = w(𝓁+1)ᵀ·δ(𝓁+1) from the layer already visited, which is why
   * the traversal runs backwards. Needs a loaded sample and a prior
   * {@link Network.feedForward}.
   *
   * @returns One {@link GradientLayer} per trainable layer, output layer first.
   * Layers without an activation function are treated as linear, with σ' = 1.
   *
   * @example
   * network.loadSample([0, 1], 1)
   * network.feedForward()
   * const gradient = network.calculateGradient()
   * gradient[0].𝓁 // 2 (the output layer comes first)
   */
  calculateGradient(): Gradient {
    const gradient: Gradient = []

    // [/doc_img/network.ts/2026-10-19-09-55-27.png]
    // [/doc_img/network.ts/2026-10-19-09-55-43.png]
    // [/doc_img/network.ts/2026-10-21-09-46-41.png]
    for (let 𝓁 = this.𝐋; 𝓁 >= 1; 𝓁--) {
      const layer = this.layers[𝓁]

      let costByActivationPartialDerivatives: NumericVector // ∂C/∂a

      if (layer.isOutputLayer) {
        costByActivationPartialDerivatives = costDerivative(
          this.costFunction,
          this.y,
          layer.a,
        )
      } else {
        const nextLayer = layer.nextLayer

        costByActivationPartialDerivatives = multiplyMatrixByVector(
          transposeMatrix(nextLayer.w),
          nextLayer.δ,
        )
      }

      // [/doc_img/network.ts/2026-09-26-18-25-14.png]
      layer.δ = this.calculateDelta(
        layer,
        costByActivationPartialDerivatives,
      )

      const gradientlayer: GradientLayer = {
        𝓁,
        w: outerProduct(layer.δ, layer.previousLayer.a), // [/doc_img/network.ts/2026-09-26-18-21-20.png]
        b: this.useBias ? [...layer.δ] : [], // [/doc_img/network.ts/2026-09-26-18-21-43.png]
      }

      gradient.push(gradientlayer)
    }

    return gradient
  }

  /**
   * Converts ∂C/∂a(𝓁) into the error signal δ(𝓁) = ∂C/∂z(𝓁), applying the
   * activation's derivative. Element-wise activations multiply by σ'(z), while
   * softmax applies its full Jacobian because every output depends on every
   * logit.
   */
  private calculateDelta(
    layer: Layer,
    costByActivationPartialDerivatives: NumericVector,
  ): NumericVector {
    if (layer.σ !== 'softmax')
      return hadamardProduct(
        costByActivationPartialDerivatives,
        layer.z.map(layer.σDerivativeFn),
      )

    // For softmax composed with cross-entropy, the Jacobian simplifies to
    // a·Σy - y. Using that form also avoids dividing by a probability that may
    // have underflowed to zero.
    if (layer.isOutputLayer && this.costFunction === 'cross-entropy') {
      const targetSum = this.y.reduce((sum, value) => sum + value, 0)

      return layer.a.map(
        (activation, i) => activation * targetSum - this.y[i],
      )
    }

    // δᵢ = aᵢ·(∂C/∂aᵢ - Σⱼ ∂C/∂aⱼ·aⱼ), the softmax Jacobian applied to ∂C/∂a.
    const weightedSum = layer.a.reduce(
      (sum, activation, i) =>
        sum + activation * costByActivationPartialDerivatives[i],
      0,
    )

    return layer.a.map(
      (activation, i) =>
        activation * (costByActivationPartialDerivatives[i] - weightedSum),
    )
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
   *
   * @example
   * network.loadSample([0, 1], 1)
   * network.feedForward()
   * const before = network.cost
   * network.backPropagate(network.calculateGradient())
   * network.feedForward()
   * // with a suitable η, the cost is now lower
   */
  backPropagate(gradient: Gradient): void {
    for (const { 𝓁, b, w } of gradient) {
      const layer = this.layers[𝓁]

      if (layer === undefined) throw new Error(`Layer ${𝓁} not found`)

      layer.w = addMatrices(
        layer.w,
        multiplyMatrixByScalar(w, -this.η * this.batchSize),
      )

      if (this.useBias && b && b.length > 0) {
        if (!layer.b || layer.b.length === 0) layer.b = Array(b.length).fill(0)

        layer.b = addVectors(
          layer.b,
          multiplyVectorByScalar(b, -this.η * this.batchSize),
        )
      }
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
   *
   * @example
   * const batch = XOR_DATA.inputs.map((inputs, i) => {
   *   network.loadSample(inputs, XOR_DATA.labels[i])
   *   network.feedForward()
   *   return network.calculateGradient()
   * })
   * const meanGradient = network.calculateAverageGradient(batch)
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
      const summedW = createMatrix(w.length, w[0]?.length ?? 0, 0)
      const summedB = b && b.length > 0 ? createVector(b.length, 0) : []

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

        if (summedB.length > 0 && layer.b && layer.b.length > 0) {
          for (let i = 0; i < summedB.length; i++) summedB[i] += layer.b[i]
        }
      }

      const averageGradientLayer: GradientLayer = {
        𝓁,
        w: divideMatrixByScalar(summedW, gradients.length),
        b:
          summedB.length > 0
            ? divideVectorByScalar(summedB, gradients.length)
            : [],
      }

      averageGradient.push(averageGradientLayer)
    })

    return averageGradient
  }

  /**
   * Saves the network's weights and biases to disk, ordered by layer, for later
   * inspection or reuse. The input layer has no parameters and is omitted.
   *
   * @param filename The filename to save the network to.
   *
   * @example
   * await network.saveToDisk('xor_network.json')
   * // writes data/xor_network.json
   */
  async saveToDisk(filename: string) {
    await Deno.writeTextFile(
      `${import.meta.dirname}/../data/${filename}`,
      JSON.stringify(
        this.layers.slice(1).map(layer => ({
          𝓁: layer.𝓁,
          w: layer.w,
          b: this.useBias && layer.b ? layer.b : [],
        })),
        null,
        2,
      ),
    )
  }

  /**
   * Loads weights and biases previously written by {@link Network.saveToDisk},
   * replacing the parameters of the matching layers. The file only supplies the
   * parameters; the topology is still the one the network was built with, so
   * the two must agree.
   *
   * @param filename The filename to load the network from.
   * @returns Nothing; each matching layer's parameters are replaced in place.
   * @throws If the file is missing, or if a saved layer or parameter shape does
   * not line up with this network, or if a layer is missing or duplicated.
   *
   * @example
   * await network.loadFromDisk('xor_network.json')
   * // reads data/xor_network.json
   */
  async loadFromDisk(filename: string): Promise<void> {
    const savedLayers = JSON.parse(
      await Deno.readTextFile(`${import.meta.dirname}/../data/${filename}`),
    ) as NetworkLayersInfo

    const loadedLayers = new Set<number>()
    const parameters: NetworkParameters = []

    for (const { 𝓁, w, b } of savedLayers) {
      if (loadedLayers.has(𝓁))
        throw new Error(`Duplicate parameters for layer ${𝓁}`)

      const layer = this.layers[𝓁]

      if (layer === undefined) throw new Error(`Layer ${𝓁} not found`)

      if (layer.isInputLayer)
        throw new Error(`Cannot load parameters for input layer ${𝓁}`)

      const previousSize = layer.previousLayer.size

      if (w.length !== layer.size || w.some(row => row.length !== previousSize))
        throw new Error(
          `Expected weights for layer ${𝓁} to be ${layer.size}x${previousSize} but got ${w.length}x${w[0]?.length ?? 0}`,
        )

      if (this.useBias) {
        if (!b || b.length !== layer.size)
          throw new Error(
            `Expected biases for layer ${𝓁} to have ${layer.size} values but got ${b?.length ?? 0}`,
          )
      }

      parameters.push({ layer, w, b: b || [] })
      loadedLayers.add(𝓁)
    }

    for (let 𝓁 = 1; 𝓁 <= this.𝐋; 𝓁++)
      if (!loadedLayers.has(𝓁))
        throw new Error(`Missing parameters for layer ${𝓁}`)

    for (const { layer, w, b } of parameters) {
      layer.w = w
      layer.b = this.useBias ? b : []
    }
  }
}
