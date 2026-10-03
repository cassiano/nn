import { NumericVector, NumericMatrix } from './types.ts'
import { Network } from './network.ts'
import { createVector } from './utils.ts'
import {
  random,
  fromMatrix,
  addMatrices,
  multiplyMatrices,
  toMatrix,
  createMatrix,
} from './utils.ts'
import {
  sigmoidDerivative,
  reluDerivative,
  tanhDerivative,
  relu,
  tanh,
  softmax,
  sigmoid,
} from './activation.ts'

/** Identifies the activation a layer applies, and so the derivative it needs during backpropagation. */
export type ActivationFunctionType = 'sigmoid' | 'relu' | 'tanh' | 'softmax'

/**
 * One layer of a {@link Network}: a group of `size` neurons holding its
 * parameters (w, b) and the values the passes compute from them (z, a, δ).
 *
 * The input layer is the exception: it only stores the raw inputs in `a` and
 * has no parameters or activation.
 */
export class Layer {
  /** Position within the network: 0 for the input layer, up to 𝐋 for the output layer. */
  𝓁: number

  /** Activations a(𝓁), the raw inputs on layer 0. */
  a: NumericVector = []

  /** Weights w(𝓁), shaped [size][previous layer size]. */
  w: NumericMatrix = []

  /** Biases b(𝓁), one per neuron. */
  b: NumericVector = []

  /** Pre-activations z(𝓁), before σ is applied. */
  z: NumericVector = []

  /** Error signal δ(𝓁) = ∂C/∂a(𝓁); empty until backpropagation runs. */
  δ: NumericVector = [] // [/doc_img/layer.ts/2026-09-26-14-25-34.png]

  /** Counts layers as they are constructed, and backs {@link Layer.𝓁}. */
  static id = 0

  /**
   * Builds a layer, randomly initializing its parameters unless it is the input
   * layer. Layers are expected to be constructed in feedforward order, since
   * each one needs the layer before it.
   *
   * @param network The network this layer belongs to.
   * @param name Human-readable label, used in debug output.
   * @param size Number of neurons.
   * @param σ Activation applied to z to produce a.
   * @returns Nothing; the layer registers itself with its network.
   * @throws If no previous layer exists yet to size w against.
   */
  constructor(
    public network: Network,
    public name: string,
    public size: number,
    public σ?: ActivationFunctionType,
  ) {
    this.𝓁 = Layer.id++

    if (!this.isInputLayer) this.initializeNetworkParameters()
  }

  /** How many parameters this layer owns: its weights plus its biases. */
  get parameterCount(): number {
    return this.isInputLayer
      ? 0
      : this.w.length * this.w[0].length + this.b.length
  }

  /** The layer feeding into this one. */
  get previousLayer(): Layer {
    return this.network.previousLayer(this.𝓁)
  }

  /** The layer fed by this one, which backpropagation uses to carry δ further back. */
  get nextLayer(): Layer {
    return this.network.nextLayer(this.𝓁)
  }

  /** Whether this is the network's first layer. */
  get isInputLayer(): boolean {
    return this.𝓁 === 0
  }

  /** Whether this is the network's last layer, whose activations are the prediction. */
  get isOutputLayer(): boolean {
    return this.𝓁 === this.network.𝐋
  }

  /**
   * Recomputes this layer's activations from its inputs: z from the previous
   * layer, then a by applying σ.
   *
   * @returns Nothing; {@link Layer.z} and {@link Layer.a} are updated in place.
   * @throws If σ is missing, or if the incoming dimensions do not line up.
   */
  calculatePostActivationValues(): void {
    this.calculatePreActivationValues()
    this.applyActivationFunction()
  }

  /**
   * The derivative σ' of this layer's activation, as a function of z, used to
   * build δ. Softmax reports a constant 1, leaving δ as the plain ∂C/∂a.
   *
   * @returns The derivative of this layer's activation.
   * @throws If σ is missing.
   */
  get σDerivativeFn(): (z: number) => number {
    if (this.σ === undefined)
      throw new Error(`Expected activation function to be defined`)

    switch (this.σ) {
      case 'sigmoid':
        return sigmoidDerivative
      case 'relu':
        return reluDerivative
      case 'tanh':
        return tanhDerivative
      case 'softmax':
        return () => 1
      default: {
        const exhaustiveCheck: never = this.σ
        throw exhaustiveCheck
      }
    }
  }

  /**
   * Gives the layer a random, non-symmetric starting point.
   *
   * @returns Nothing; {@link Layer.w} and {@link Layer.b} are set in place.
   */
  private initializeNetworkParameters(): void {
    this.w = createMatrix(this.size, this.previousLayer.size, () =>
      random(-1, 1),
    )
    this.b = createVector(this.size, () => random(-1, 1))
  }

  /**
   * Computes z(𝓁) = w(𝓁)·a(𝓁-1) + b(𝓁).
   *
   * @returns Nothing; {@link Layer.z} is stored in place.
   * @throws If w, b and the previous layer's `a` disagree on their dimensions.
   */
  private calculatePreActivationValues(): void {
    this.z = fromMatrix(
      addMatrices(
        multiplyMatrices(this.w, toMatrix(this.previousLayer.a)),
        toMatrix(this.b),
      ),
    )
  }

  /**
   * Applies σ to z, storing the result in a.
   *
   * @returns Nothing; {@link Layer.a} is stored in place.
   * @throws If σ is missing.
   */
  private applyActivationFunction(): void {
    if (this.σ === undefined)
      throw new Error(`Expected activation function to be defined`)

    switch (this.σ) {
      case 'sigmoid':
        this.a = this.z.map(sigmoid)
        break
      case 'relu':
        this.a = this.z.map(relu)
        break
      case 'tanh':
        this.a = this.z.map(tanh)
        break
      case 'softmax':
        this.a = softmax(this.z)
        break
      default: {
        const exhaustiveCheck: never = this.σ
        throw exhaustiveCheck
      }
    }
  }
}
