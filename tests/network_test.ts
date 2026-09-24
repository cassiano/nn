import { sigmoid, sigmoidDerivative } from '../activation.ts'
import { makeStandardNetwork, makeTinyNetwork, resetLayerCounter } from './test_helpers.ts'
import {
  assert,
  assertEquals,
  assertThrows,
  assertClose,
  assertArrayClose,
} from './test_helpers.ts'
import { Network } from '../network.ts'
import { MNIST_IMAGE_COLS, MNIST_IMAGE_ROWS, MNIST_OUTPUT_SIZE } from '../mnist_loader.ts'

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
  const net = makeTinyNetwork()
  const [input, hidden, output] = net.layers

  input.a = [0.5, 0.9]
  hidden.w = [[0.7, -0.2], [0.1, 0.4]]
  hidden.b = [0.1, -0.3]
  output.w = [[1, 0.5], [-0.5, 2]]
  output.b = [0.2, -0.1]
  net.y = [1, 0]
  net.feedForward()

  const eps = 1e-5
  const grad = output.calculateGradient()

  const reconstruct = (weight: number, biasOffset = 0) => {
    // Perturb w[0][0] and, if requested, b[0], then measure the true cost.
    const originalW = output.w[0][0]
    const originalB = output.b[0]
    output.w[0][0] = weight
    output.b[0] = output.b[0] + biasOffset
    net.feedForward()
    const value = net.cost
    output.w[0][0] = originalW
    output.b[0] = originalB
    return value
  }

  // ∂C/∂w[0][0]: quadratic cost is smooth, so central differences are accurate.
  const numericDwdW = (reconstruct(output.w[0][0] + eps) - reconstruct(output.w[0][0] - eps)) / (2 * eps)
  assertClose(grad[0], numericDwdW, 1e-4)

  // ∂C/∂b[0] is the entry right after all weight partials (2 weights/row * 2 rows).
  const numericDwdB = (reconstruct(output.w[0][0], eps) - reconstruct(output.w[0][0], -eps)) / (2 * eps)
  assertClose(grad[4], numericDwdB, 1e-4)
})

Deno.test('Network / inner layer gradient follows the implemented non-backprop rule', () => {
  const net = makeTinyNetwork()
  const [input, hidden, output] = net.layers

  input.a = [0.5, 0.9]
  hidden.w = [[0.7, -0.2], [0.1, 0.4]]
  hidden.b = [0.1, -0.3]
  output.w = [[1, 0.5], [-0.5, 2]]
  output.b = [0.2, -0.1]
  net.y = [1, 0]
  net.feedForward()

  const grad = hidden.calculateGradient()
  // Current (placeholder) rule, before full backpropagation is implemented:
  // ∂C/∂w[h][r][c] = a_prev[c] · σ'(z[r]) · 2·(a[r] - y[r])
  const expectedW00 = input.a[0] * sigmoidDerivative(hidden.z[0]) * 2 * (hidden.a[0] - net.y[0])
  assertClose(grad[0], expectedW00, 1e-12)

  // Biases live after the weight partials: index = rows*cols.
  const expectedB0 = 1 * sigmoidDerivative(hidden.z[0]) * 2 * (hidden.a[0] - net.y[0])
  assertClose(grad[2 * 2], expectedB0, 1e-12)
})

Deno.test('Network / calculateLayerGradient delegates to the layer', () => {
  const net = makeTinyNetwork()
  const [input] = net.layers
  input.a = [0.5, 0.9]
  net.y = [1, 0]
  net.feedForward()

  assertEquals(net.calculateLayerGradient(2), net.layers[2].calculateGradient())
})