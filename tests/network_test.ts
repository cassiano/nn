import { sigmoid } from '../activation.ts'
import { makeStandardNetwork, makeTinyNetwork, makeTinyNetworkWithoutActivation, resetLayerCounter } from './test_helpers.ts'
import {
  assert,
  assertEquals,
  assertThrows,
  assertClose,
  assertArrayClose,
} from './test_helpers.ts'
import { Network } from '../network.ts'
import { Layer } from '../layer.ts'
import { MNIST_IMAGE_COLS, MNIST_IMAGE_ROWS, MNIST_OUTPUT_SIZE } from '../mnist_loader.ts'

/**
 * Builds a 2 -> 2 -> 2 network with fixed weights, so the gradient can be
 * checked against central differences on hand-picked numbers.
 */
function makeSeededTinyNetwork(): Network {
  const net = makeTinyNetwork()
  const [input, hidden, output] = net.layers

  input.a = [0.5, 0.9]
  hidden.w = [[0.7, -0.2], [0.1, 0.4]]
  hidden.b = [0.1, -0.3]
  output.w = [[1, 0.5], [-0.5, 2]]
  output.b = [0.2, -0.1]
  net.y = [1, 0]
  net.feedForward()

  return net
}

/**
 * Approximates ∂C/∂(w[row][col] or b[row]) by central differences, perturbing
 * one parameter, re-running the forward pass and reading the true cost. The
 * quadratic cost is smooth, so central differences are accurate to ~1e-9 for
 * the step size used here. The network is left exactly as it was found.
 *
 * @param net A network whose sample is already loaded.
 * @param layer The layer owning the parameter to perturb.
 * @param row Row index of the weight or bias entry.
 * @param col Column index of the weight entry, or null to perturb the bias.
 * @param delta Half the perturbation step.
 * @returns The finite-difference partial derivative.
 */
function numericPartial(
  net: Network,
  layer: Layer,
  row: number,
  col: number | null,
  delta = 1e-5,
): number {
  const originalW = col === null ? 0 : layer.w[row][col]
  const originalB = layer.b[row]

  const set = (offset: number) => {
    if (col === null) layer.b[row] = originalB + offset
    else layer.w[row][col] = originalW + offset

    net.feedForward()

    return net.cost
  }

  const perturbed = set(delta)
  const base = set(-delta)

  if (col === null) layer.b[row] = originalB
  else layer.w[row][col] = originalW

  net.feedForward()

  return (perturbed - base) / (2 * delta)
}

Deno.test('Network / constructor builds the expected topology', () => {
  const net = makeStandardNetwork()
  assertEquals(net.layers.length, 4)
  assertEquals(net.inputLayer.size, MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
  assertEquals(net.outputLayer.size, MNIST_OUTPUT_SIZE)
})

Deno.test('Network / constructor rejects a non-784 input layer', () => {
  resetLayerCounter()
  assertThrows(
    () =>
      new Network(
        [{ name: 'in', size: 1 }, { name: 'out', size: 10 }],
        0.01,
      ),
    'Expected input layer size',
  )
})

Deno.test('Network / constructor rejects a non-10-class output layer', () => {
  resetLayerCounter()
  assertThrows(
    () =>
      new Network(
        [{ name: 'in', size: 784 }, { name: 'out', size: 3 }],
        0.01,
      ),
    'Expected output layer size',
  )
})

Deno.test('Network / parameterCount matches the hand calculation', () => {
  const net = makeStandardNetwork()
  const expected = 784 * 16 + 16 + 16 * 16 + 16 + 16 * 10 + 10
  assertEquals(net.parameterCount, expected)
  assertEquals(net.parameterCount, 13002)
  // The input layer itself has no learnable parameters.
  assertEquals(net.inputLayer.parameterCount, 0)
  // The whole network is just the sum of its layers.
  assertEquals(net.parameterCount, net.layers.reduce((s, l) => s + l.parameterCount, 0))
})

Deno.test('Network / constructor initializes all parameters randomly in [-1, 1)', () => {
  const net = makeStandardNetwork()
  for (const layer of net.layers.slice(1)) {
    for (const row of layer.w)
      for (const value of row) assert(value >= -1 && value < 1)
    for (const value of layer.b) assert(value >= -1 && value < 1)
  }
})

Deno.test('Network / loadSample stores inputs and one-hot-encodes the label', () => {
  const net = makeStandardNetwork()
  const inputs = Array.from({ length: 784 }, (_, i) => i / 784)
  net.loadSample(inputs, 4)
  assertEquals(net.inputLayer.a, inputs)
  assertEquals(net.y, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
})

Deno.test('Network / loadSample rejects inputs of the wrong length', () => {
  const net = makeStandardNetwork()
  assertThrows(() => net.loadSample([1, 2, 3], 0), 'Expected input size')
})

Deno.test('Network / cost is zero for a perfect prediction', () => {
  const net = makeStandardNetwork()
  net.loadSample(new Array(784).fill(0.5), 3)
  net.outputLayer.a = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  assertEquals(net.cost, 0)
})

Deno.test('Network / cost sums the squared errors against the target', () => {
  const net = makeStandardNetwork()
  net.loadSample(new Array(784).fill(0), 0) // y = [1, 0, 0, ...]
  const a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  net.outputLayer.a = a
  const expected = a.reduce((sum, value, i) => sum + (net.y[i] - value) ** 2, 0)
  assertClose(net.cost, expected, 1e-12)
})

Deno.test('Network / cost tracks manual values on a tiny network', () => {
  const net = makeTinyNetwork()
  net.outputLayer.a = [0.25, 0.75]
  net.y = [1, 0]
  assertClose(net.cost, (1 - 0.25) ** 2 + (0 - 0.75) ** 2, 1e-12)
})

Deno.test('Network / feedForward propagates hand-computable values', () => {
  const net = makeTinyNetwork()
  const [input, hidden, output] = net.layers

  input.a = [0.5, 0.9]
  hidden.w = [[0.7, -0.2], [0.1, 0.4]]
  hidden.b = [0.1, -0.3]
  output.w = [[1, 0.5], [-0.5, 2]]
  output.b = [0.2, -0.1]

  net.feedForward()

  // Hidden layer: z1[0] = 0.7*0.5 + (-0.2)*0.9 + 0.1, z1[1] = 0.1*0.5 + 0.4*0.9 - 0.3
  const z1 = [
    0.7 * 0.5 + -0.2 * 0.9 + 0.1,
    0.1 * 0.5 + 0.4 * 0.9 + -0.3,
  ]
  assertArrayClose(hidden.z, z1, 1e-12)
  assertArrayClose(hidden.a, z1.map(sigmoid), 1e-12)

  // Output layer: z2 = w·a(hidden) + b
  const z2 = [
    output.w[0][0] * hidden.a[0] + output.w[0][1] * hidden.a[1] + output.b[0],
    output.w[1][0] * hidden.a[0] + output.w[1][1] * hidden.a[1] + output.b[1],
  ]
  assertArrayClose(output.z, z2, 1e-12)
  assertArrayClose(output.a, z2.map(sigmoid), 1e-12)
})

Deno.test('Network / output layer gradient matches numerical differentiation', () => {
  const net = makeSeededTinyNetwork()
  const [, , output] = net.layers

  // calculateGradient() orders entries output layer first, so index 0 is 𝐋.
  const { w, b } = net.calculateGradient()[0]

  assertClose(w[0][0], numericPartial(net, output, 0, 0), 1e-4)
  assertClose(w[1][1], numericPartial(net, output, 1, 1), 1e-4)
  assertClose(b[0], numericPartial(net, output, 0, null), 1e-4)
  assertClose(b[1], numericPartial(net, output, 1, null), 1e-4)
})

Deno.test('Network / inner layer gradient matches numerical differentiation', () => {
  const net = makeSeededTinyNetwork()
  const [, hidden] = net.layers

  // The hidden layer's entry sits after the output layer's, at index 1.
  const { 𝓁, w, b } = net.calculateGradient()[1]
  assertEquals(𝓁, 1)

  // The error reaching the hidden layer is propagated from the output layer
  // through w(2)ᵀ · δ(2), not read straight off y, so its partials must still
  // agree with central differences. This is the regression guard for the
  // placeholder rule that indexed y by neuron and produced NaN.
  assertClose(w[0][0], numericPartial(net, hidden, 0, 0), 1e-4)
  assertClose(w[1][1], numericPartial(net, hidden, 1, 1), 1e-4)
  assertClose(b[0], numericPartial(net, hidden, 0, null), 1e-4)
  assertClose(b[1], numericPartial(net, hidden, 1, null), 1e-4)
})

Deno.test('Network / calculateGradient covers every trainable layer once', () => {
  const net = makeTinyNetwork()
  const [input] = net.layers
  input.a = [0.5, 0.9]
  net.y = [1, 0]
  net.feedForward()

  const gradient = net.calculateGradient()

  // One entry per layer that owns parameters; the input layer has none.
  assertEquals(gradient.length, net.layers.length - 1)
  // Ordered output layer first, then each inner layer descending to 1.
  assertEquals(gradient.map(entry => entry.𝓁), [2, 1])

  // Each entry mirrors the shape of the parameters it belongs to.
  for (const { 𝓁, w, b } of gradient) {
    const layer = net.layers[𝓁]
    assertEquals(w.length, layer.w.length)
    assertEquals(w[0].length, layer.w[0].length)
    assertEquals(b.length, layer.b.length)
  }
})

Deno.test('Network / gradient holds one partial per trainable parameter', () => {
  // Each layer contributes ∂C/∂w for every weight and ∂C/∂b for every bias,
  // so the flattened partial count must equal the network's parameterCount.
  const net = makeStandardNetwork()
  net.loadSample(new Array(784).fill(0.5), 3)
  net.feedForward()

  const gradient = net.calculateGradient()
  const partialCount = gradient.reduce(
    (sum, { w, b }) => sum + w.length * w[0].length + b.length,
    0,
  )

  assertEquals(partialCount, net.parameterCount)
  assertEquals(net.parameterCount, 13002)
})

Deno.test('Network / gradient partials are finite and defined', () => {
  // The error term is propagated backwards through w(𝓁+1)ᵀ · δ(𝓁+1), so no
  // layer may read past the end of y. This is the regression guard for the
  // placeholder rule that produced NaN across the 16-neuron hidden layers.
  const net = makeStandardNetwork()
  net.loadSample(new Array(784).fill(0.5), 3)
  net.feedForward()

  for (const { w, b } of net.calculateGradient()) {
    for (const partial of b) assert(Number.isFinite(partial))
    for (const row of w) for (const partial of row) assert(Number.isFinite(partial))
  }
})

Deno.test('Network / calculateGradient throws when a layer has no activation', () => {
  // σDerivativeFn is read for every layer, so a missing σ is rejected whether it
  // sits on the output layer or an inner one. Deliberately no feedForward() call:
  // the forward pass would reject the same missing σ first, and this targets the
  // gradient's own error path.
  //
  // Note makeTinyNetworkWithoutActivation's σ argument defaults to 'sigmoid', so
  // passing undefined would just select that default; σ is cleared afterwards.
  const withoutOutput = makeTinyNetworkWithoutActivation()
  withoutOutput.layers[0].a = [1, 1]
  withoutOutput.outputLayer.σ = undefined
  withoutOutput.y = [1, 0]
  assertThrows(
    () => withoutOutput.calculateGradient(),
    'Expected activation function to be defined',
  )

  // Here the output layer is fine, so the walk has to reach the σ-less hidden
  // layer. Its a is filled in by hand along with the output layer's z and a, so
  // the output layer's own partials can be computed before the failure.
  const withoutHidden = makeTinyNetworkWithoutActivation()
  const [input, hidden, output] = withoutHidden.layers
  input.a = [1, 1]
  hidden.a = [0.5, 0.5]
  output.z = [0.5, -0.5]
  output.a = [0.6, 0.4]
  withoutHidden.y = [1, 0]
  assertThrows(
    () => withoutHidden.calculateGradient(),
    'Expected activation function to be defined',
  )
})
