import { Network } from '../network.ts'
import { Layer, ActivationFunctionType } from '../layer.ts'

/** Resets the static `Layer.id` counter, so layer indices (𝓁) are deterministic. */
export function resetLayerCounter(): void {
  Layer.id = 0
}

/** The standard MNIST-sized network (784 -> 16 -> 16 -> 10), with random weights. */
export function makeStandardNetwork(): Network {
  resetLayerCounter()
  return new Network(
    [
      { name: 'Input Layer', size: 784 },
      { name: 'Hidden Layer 1', size: 16, σ: 'sigmoid' },
      { name: 'Hidden Layer 2', size: 16, σ: 'relu' },
      { name: 'Output Layer', size: 10, σ: 'softmax' },
    ],
    0.01,
  )
}

/**
 * A small 2 -> 2 -> 2 network, built without the constructor's MNIST size
 * validation so tests can hand-verify the math on concrete numbers.
 */
export function makeTinyNetwork(
  σHidden: ActivationFunctionType = 'sigmoid',
  σOutput: ActivationFunctionType = 'sigmoid',
): Network {
  resetLayerCounter()
  const network = Object.create(Network.prototype) as Network
  network.layers = []
  network.y = []
  network.η = 0.01

  network.layers.push(new Layer(network, 'input', 2))
  network.layers.push(new Layer(network, 'hidden', 2, σHidden))
  network.layers.push(new Layer(network, 'output', 2, σOutput))

  return network
}

/** A tiny network whose hidden layer has no σ, for the "activation required" paths. */
export function makeTinyNetworkWithoutActivation(
  σOutput: ActivationFunctionType | undefined = 'sigmoid',
): Network {
  resetLayerCounter()
  const network = Object.create(Network.prototype) as Network
  network.layers = []
  network.y = []
  network.η = 0.01

  network.layers.push(new Layer(network, 'input', 2))
  network.layers.push(new Layer(network, 'hidden', 2))
  network.layers.push(new Layer(network, 'output', 2, σOutput))

  return network
}

/** Asserts a truthy condition, throwing `message` otherwise. */
export function assert(
  condition: unknown,
  message = 'Assertion failed',
): asserts condition {
  if (!condition) throw new Error(message)
}

/** Asserts deep equality, comparing arrays element-wise and objects by key. */
export function assertEquals(
  actual: unknown,
  expected: unknown,
  message = '',
): void {
  if (!deepEqual(actual, expected)) {
    const prefix = message ? `${message}: ` : ''
    throw new Error(
      `${prefix}Expected ${fmt(actual)} to equal ${fmt(expected)}`,
    )
  }
}

/**
 * Asserts that `actual` is within `eps` of `expected`, for float comparisons.
 * A non-finite `actual` is rejected, since it would pass the comparison
 * vacuously.
 */
export function assertClose(
  actual: number,
  expected: number,
  eps = 1e-9,
  message = '',
): void {
  if (!Number.isFinite(actual))
    throw new Error(
      `${message}Expected ${actual} to be a finite number close to ${expected}`,
    )

  if (Math.abs(actual - expected) > eps)
    throw new Error(
      `${message}Expected ${actual} to be within ${eps} of ${expected}`,
    )
}

/** Asserts that two numeric arrays are element-wise within `eps`. */
export function assertArrayClose(
  actual: number[],
  expected: number[],
  eps = 1e-9,
  message = '',
): void {
  assert(
    actual.length === expected.length,
    `${message}Lengths differ: ${actual.length} vs ${expected.length}`,
  )
  for (let i = 0; i < actual.length; i++)
    assertClose(actual[i], expected[i], eps, message)
}

/**
 * Asserts that `fn` throws, and that the message contains `match` (a substring
 * or RegExp).
 */
export function assertThrows(fn: () => void, match: string | RegExp): void {
  try {
    fn()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    const ok = match instanceof RegExp ? match.test(message) : message.includes(match)
    assert(
      ok,
      `Expected thrown error ${JSON.stringify(message)} to match ${String(match)}`,
    )
    return
  }
  throw new Error('Expected function to throw, but it completed')
}

/** Structural equality behind {@link assertEquals}, recursing into arrays and objects. */
function deepEqual(actual: unknown, expected: unknown): boolean {
  if (Object.is(actual, expected)) return true
  if (typeof actual !== typeof expected) return false
  if (Array.isArray(actual) && Array.isArray(expected)) {
    if (actual.length !== expected.length) return false
    return actual.every((value, index) => deepEqual(value, expected[index]))
  }
  if (
    actual !== null && expected !== null &&
    typeof actual === 'object' && typeof expected === 'object'
  ) {
    const actualKeys = Object.keys(actual as object)
    const expectedKeys = Object.keys(expected as object)
    if (actualKeys.length !== expectedKeys.length) return false
    return actualKeys.every(key =>
      deepEqual(
        (actual as Record<string, unknown>)[key],
        (expected as Record<string, unknown>)[key],
      )
    )
  }
  return false
}

/** Renders a value for a failure message, falling back to `String(value)`. */
function fmt(value: unknown): string {
  return JSON.stringify(value) ?? String(value)
}
