import { Network } from '../network.ts'
import { Layer, ActivationFunctionType } from '../layer.ts'

/**
 * Resets the static `Layer.id` counter so that layer indices (𝓁) are
 * deterministic within a test. Call before building any network.
 */
export function resetLayerCounter(): void {
  Layer.id = 0
}

/**
 * Builds the standard MNIST-sized network (784 -> 16 -> 16 -> 10) through the
 * real constructor, exercising its size validation and weight initialization.
 */
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
 * Builds a small 2 -> 2 -> 2 network by bypassing the MNIST-specific
 * constructor validation (which requires 784 inputs / 10 outputs), so tests
 * can hand-verify the math on tiny, concrete numbers.
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

/** Asserts a truthy condition, throwing with `message` otherwise. */
export function assert(
  condition: unknown,
  message = 'Assertion failed',
): asserts condition {
  if (!condition) throw new Error(message)
}

/**
 * Asserts deep equality between two values (numbers, strings, nested arrays
 * and plain objects).
 */
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

/** Asserts that `actual` is within `eps` of `expected`. */
export function assertClose(
  actual: number,
  expected: number,
  eps = 1e-9,
  message = '',
): void {
  if (Math.abs(actual - expected) > eps)
    throw new Error(
      `${message}Expected ${actual} to be within ${eps} of ${expected}`,
    )
}

/** Asserts element-wise closeness of two numeric arrays. */
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
 * Asserts that `fn` throws, and that the error message contains / matches `match`.
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

function fmt(value: unknown): string {
  return JSON.stringify(value) ?? String(value)
}