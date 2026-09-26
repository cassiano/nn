import { NumericVector, NumericMatrix } from './types.ts'
import { Network } from './network.ts'
import {
  random,
  timesMap,
  timesMapN,
  fromMatrix,
  addMatrices,
  multiplyMatrices,
  toMatrix,
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

/**
 * Identifies which activation function a layer applies, and therefore which
 * derivative {@link Layer.σDerivativeFn} returns.
 * - 'sigmoid': logistic, outputs (0, 1)
 * - 'relu': max(0, x), outputs [0, ∞)
 * - 'tanh': outputs (-1, 1)
 * - 'softmax': probability distribution across the layer, outputs sum to 1
 */
export type ActivationFunctionType = 'sigmoid' | 'relu' | 'tanh' | 'softmax'

/**
 * A single layer of a {@link Network}: a fixed-size group of neurons.
 *
 * Stores the layer's trainable parameters and intermediate values:
 * - w: weight matrix ([size][previous layer size]), random in (-1, 1) at init
 * - b: bias vector ([size]), random in (-1, 1) at init
 * - z: pre-activation values z(𝓁) = w(𝓁)·a(𝓁-1) + b(𝓁)
 * - a: post-activation values, obtained by applying σ to z
 * - δ: error signal δ(𝓁) = ∂C/∂a(𝓁), written only by backpropagation
 *
 * The first (input) layer is a special case: it only stores the raw inputs in
 * `a` and completely ignores w, b, z, δ and σ.
 */
export class Layer {
  /**
   * Position of this layer within its network: 0 for the input layer, up to 𝐋
   * for the output layer. Assigned from the static {@link Layer.id} counter
   * rather than passed in, so a network's layers get consecutive indices in
   * construction order.
   */
  𝓁: number

  /** Post-activation values a(𝓁) ([size]); the raw inputs on layer 0. */
  a: NumericVector = []

  /** Weights w(𝓁) ([size][previous layer size]); empty on the input layer. */
  w: NumericMatrix = []

  /** Biases b(𝓁) ([size]); empty on the input layer. */
  b: NumericVector = []

  /**
   * Pre-activation values z(𝓁) ([size]), i.e. the weighted sum of the previous
   * layer's activations plus this layer's biases, before σ is applied.
   */
  z: NumericVector = []

  /**
   * Error signal δ(𝓁) ([size]) = ∂C/∂a(𝓁), how much this layer's activations
   * contributed to the cost. Produced by
   * {@link Network.calculateGradient} while walking layers backwards, and read
   * by the layer after this one; empty until backpropagation runs.
   * [/doc_img/layer.ts/2026-09-26-14-25-34.png]
   */
  δ: NumericVector = []

  /**
   * Monotonic counter backing {@link Layer.𝓁}. Tests reset it (e.g. via
   * `resetLayerCounter()`) so layer indices are deterministic.
   */
  static id = 0

  /**
   * Builds a layer and, unless it is the input layer, gives it randomly
   * initialized weights and biases. A layer is expected to be constructed in
   * feedforward order, because its index and its weight shape both depend on
   * the layer before it.
   *
   * @param network The network this layer belongs to; used to look up
   * {@link Layer.previousLayer} during initialization.
   * @param name Human-readable label for this layer, used in debug output.
   * @param size Number of neurons, i.e. the length of a, z, δ and b.
   * @param σ Activation function (greek letter sigma) applied to z to produce
   * a. Omit it for the input layer, which has no weights and applies no
   * activation.
   */
  constructor(
    public network: Network,
    public name: string,
    public size: number,
    public σ?: ActivationFunctionType, // Activation function (greek letter sigma)
  ) {
    this.𝓁 = Layer.id++

    if (this.𝓁 > 0) this.initializeNetworkParameters()
  }

  /**
   * Number of learnable parameters owned by this layer: every weight plus
   * every bias. The input layer reports 0, since it has none.
   */
  get parameterCount() {
    return this.𝓁 === 0 ? 0 : this.w.length * this.w[0].length + this.b.length
  }

  /**
   * The layer whose activations feed into this one, i.e. the network's layer at
   * index `𝓁 - 1`.
   *
   * @throws If this is the input layer, which has no predecessor.
   */
  get previousLayer() {
    return this.network.previousLayer(this.𝓁)
  }

  get nextLayer() {
    return this.network.nextLayer(this.𝓁)
  }

  get isInputLayer() {
    return this.𝓁 === 0
  }

  get isOutputLayer() {
    return this.𝓁 === this.network.𝐋
  }

  /**
   * Recomputes this layer's activations from its incoming values: first
   * {@link Layer.z} from the previous layer's `a` and this layer's parameters,
   * then `a` by applying σ to z. Together these are the two halves of the
   * layer's forward pass.
   *
   * @throws If σ is not configured, or if the dimensions of w, b and the
   * previous layer's `a` do not line up.
   */
  calculatePostActivationValues() {
    this.calculatePreActivationValues()
    this.applyActivationFunction()
  }

  /**
   * The derivative σ' of this layer's activation function, as a function of
   * the pre-activation z. Needed during backpropagation to turn ∂C/∂a into
   * δ(𝓁) = ∂C/∂a · σ'(z(𝓁)).
   *
   * @returns The matching derivative function, except for 'softmax', which
   * returns a constant 1 so that δ stays the unweighted ∂C/∂a.
   * @throws If σ is not configured.
   */
  get σDerivativeFn() {
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
   * Fills w and b with random values in (-1, 1), giving the network a
   * non-symmetric starting point. Skipped for the input layer, which has no
   * incoming connections. Called only from the constructor.
   *
   * @throws If the previous layer has not been created yet, since w's shape is
   * derived from its size.
   */
  private initializeNetworkParameters() {
    this.w = timesMapN([this.size, this.previousLayer.size], () =>
      random(-1, 1),
    )
    this.b = timesMap(this.size, () => random(-1, 1))
  }

  /**
   * Computes the pre-activations z(𝓁) = w(𝓁)·a(𝓁-1) + b(𝓁) and stores them in
   * `z`. The matrix helpers convert between the column/row shapes involved: the
   * previous layer's vector becomes an Nx1 matrix, the product is folded back
   * into a single column, and the bias vector is broadcast by adding one row
   * per bias.
   *
   * @throws If w's columns, the previous layer's `a` length and b's length
   * disagree, or if the input layer calls this (it has no parameters).
   */
  private calculatePreActivationValues() {
    // z(𝓁) = w(𝓁) * a(𝓁-1) + b(𝓁)
    this.z = fromMatrix(
      addMatrices(
        multiplyMatrices(this.w, toMatrix(this.previousLayer.a)),
        toMatrix(this.b),
      ),
    )
  }

  /**
   * Applies σ to every entry of `z` and stores the result in `a`. Dispatches
   * on {@link Layer.σ}; the activation functions all apply element-wise except
   * 'softmax', which normalizes across the whole vector so its outputs sum
   * to 1.
   *
   * @throws If σ is not configured.
   */
  private applyActivationFunction() {
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
