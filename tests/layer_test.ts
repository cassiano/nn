import { sigmoid, sigmoidDerivative, reluDerivative, tanhDerivative } from '../activation.ts'
import { ActivationFunctionType } from '../layer.ts'
import {
  makeTinyNetwork,
  makeTinyNetworkWithoutActivation,
} from './test_helpers.ts'
import {
  assert,
  assertEquals,
  assertThrows,
  assertClose,
  assertArrayClose,
} from './test_helpers.ts'

Deno.test('Layer / input layer carries no trainable state', () => {
  const net = makeTinyNetwork()
  const input = net.layers[0]
  assertEquals(input.𝓁, 0)
  assertEquals(input.w, [])
  assertEquals(input.b, [])
  assertEquals(input.z, [])
  assertEquals(input.parameterCount, 0)
})

Deno.test('Layer / layers receive sequential indices', () => {
  const net = makeTinyNetwork()
  assertEquals(
    net.layers.map(layer => layer.𝓁),
    [0, 1, 2],
  )
})

Deno.test('Layer / hidden layers initialize weights and biases', () => {
  const net = makeTinyNetwork()
  const hidden = net.layers[1]
  assertEquals(hidden.𝓁, 1)
  assertEquals(hidden.size, 2)
  // weight matrix is [size][previous layer size] = 2x2
  assertEquals(hidden.w.length, 2)
  assertEquals(hidden.w[0].length, 2)
  assertEquals(hidden.b.length, 2)
})

Deno.test('Layer / parameterCount counts all weights plus all biases', () => {
  const net = makeTinyNetwork()
  assertEquals(net.layers[1].parameterCount, 2 * 2 + 2)
  // Input layer contributes nothing.
  assertEquals(net.layers[0].parameterCount, 0)
})

Deno.test(
  'Layer / weights and biases are randomly initialized in [-1, 1)',
  () => {
    const net = makeTinyNetwork()
    const hidden = net.layers[1]
    for (const row of hidden.w)
      for (const value of row) assert(value >= -1 && value < 1)
    for (const value of hidden.b) assert(value >= -1 && value < 1)
  },
)

Deno.test('Layer / calculatePostActivationValues computes z then a', () => {
  const net = makeTinyNetwork()
  const [input, hidden] = net.layers
  input.a = [1, 2]
  hidden.w = [
    [1, 0],
    [0, 1],
  ]
  hidden.b = [0.5, -0.5]

  hidden.calculatePostActivationValues()

  // z = w·a + b = [1*1 + 0*2 + 0.5, 0*1 + 1*2 - 0.5] = [1.5, 1.5]
  assertArrayClose(hidden.z, [1.5, 1.5], 1e-12)
  // a = sigmoid(z)
  assertArrayClose(hidden.a, [sigmoid(1.5), sigmoid(1.5)], 1e-12)
})

Deno.test('Layer / relu activation clamps negatives to zero', () => {
  const net = makeTinyNetwork('relu', 'relu')
  const [input, hidden] = net.layers
  input.a = [-2, 3]
  hidden.w = [
    [1, 0],
    [0, 1],
  ]
  hidden.b = [0, 0]

  hidden.calculatePostActivationValues()

  assertArrayClose(hidden.a, [0, 3], 1e-12)
})

Deno.test('Layer / softmax output produces a probability distribution', () => {
  const net = makeTinyNetwork('relu', 'softmax')
  const [input, hidden, output] = net.layers
  input.a = [0.1, 0.2]
  hidden.w = [
    [1, 0],
    [0, 1],
  ]
  hidden.b = [0, 0]
  output.w = [
    [1, 0],
    [0, 1],
  ]
  output.b = [0.3, 0.1]

  // Run the forward pass so the relu hidden layer feeds the softmax output.
  net.feedForward()

  assertClose(
    output.a.reduce((sum, v) => sum + v, 0),
    1,
    1e-12,
  )
  for (const value of output.a) assert(value > 0)
})

Deno.test(
  'Layer / forward pass throws when no activation is configured',
  () => {
    const net = makeTinyNetworkWithoutActivation()
    const [input, hidden] = net.layers
    input.a = [1, 1]
    hidden.w = [
      [1, 0],
      [0, 1],
    ]
    hidden.b = [0, 0]

    assertThrows(
      () => hidden.calculatePostActivationValues(),
      'activation function to be defined',
    )
  },
)

Deno.test('Layer / σDerivativeFn returns the derivative matching σ', () => {
  // Each activation's derivative is what turns ∂C/∂a into δ during
  // backpropagation, so the dispatch has to follow σ.
  const cases: Array<[ActivationFunctionType, (z: number) => number]> = [
    ['sigmoid', sigmoidDerivative],
    ['relu', reluDerivative],
    ['tanh', tanhDerivative],
  ]

  for (const [σ, expected] of cases) {
    const net = makeTinyNetwork(σ) // σ on the hidden layer
    const layer = net.layers[1]

    assertEquals(layer.σDerivativeFn(0.5), expected(0.5))
  }
})

Deno.test('Layer / σDerivativeFn reports a constant 1 for softmax', () => {
  // softmax has no elementwise derivative, so the output layer's δ is left as
  // the plain ∂C/∂a and the jacobian is folded into w instead.
  const net = makeTinyNetwork('relu', 'softmax')
  const output = net.outputLayer

  assertEquals(output.σDerivativeFn(0), 1)
  assertEquals(output.σDerivativeFn(-7.5), 1)
  assertEquals(output.σDerivativeFn(123), 1)
})

Deno.test('Layer / σDerivativeFn throws when σ is not configured', () => {
  const net = makeTinyNetworkWithoutActivation()

  assertThrows(
    () => net.layers[1].σDerivativeFn,
    'Expected activation function to be defined',
  )
})

Deno.test('Layer / previousLayer and nextLayer resolve the neighbours', () => {
  const net = makeTinyNetwork()
  const [input, hidden, output] = net.layers

  assertEquals(hidden.previousLayer, input)
  assertEquals(hidden.nextLayer, output)
})

Deno.test('Layer / previousLayer throws on the input layer', () => {
  const net = makeTinyNetwork()

  assertThrows(
    () => net.inputLayer.previousLayer,
    'Input layer does not have a previous one',
  )
})

Deno.test('Layer / nextLayer throws on the output layer', () => {
  const net = makeTinyNetwork()

  assertThrows(
    () => net.outputLayer.nextLayer,
    'Output layer does not have a next one',
  )
})

Deno.test('Layer / isInputLayer and isOutputLayer flag the ends of the stack', () => {
  const net = makeTinyNetwork()
  const [input, hidden, output] = net.layers

  assert(input.isInputLayer)
  assert(!input.isOutputLayer)
  assert(!hidden.isInputLayer)
  assert(!hidden.isOutputLayer)
  assert(!output.isInputLayer)
  assert(output.isOutputLayer)
})
