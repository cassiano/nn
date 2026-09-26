import { Network } from '../network.ts'
import { Layer, ActivationFunctionType } from '../layer.ts'

/**
 * Resets the static `Layer.id` counter so that layer indices (𝓁) are
 * deterministic within a test. Call before building any network.
 *
 * @returns Nothing.
 */
export function resetLayerCounter(): void {
  Layer.id = 0
}

/**
 * Builds the standard MNIST-sized network (784 -> 16 -> 16 -> 10) through the
 * real constructor, exercising its size validation and weight initialization.
 *
 * Weights and biases are randomized, so tests needing concrete values must
 * overwrite them afterwards.
 *
 * @returns A fresh network with a reset layer counter and η = 0.01.
 * @throws Never in practice, since the topology always satisfies the MNIST
 * size validation.
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
 *
 * @param σHidden Activation for the hidden layer; defaults to 'sigmoid'.
 * @param σOutput Activation for the output layer; defaults to 'sigmoid'. Use
 * 'softmax' to test probability-shaped outputs.
 * @returns A fresh network with a reset layer counter, η = 0.01, and randomly
 * initialized weights. The input layer has no σ, as expected.
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

/**
 * Builds a tiny 2 -> 2 -> 2 network whose hidden layer has no activation
 * function configured (σ omitted), for exercising the "activation required"
 * error paths.
 *
 * @param σOutput Activation for the output layer; defaults to 'sigmoid'. Pass
 * `undefined` to leave the output layer without σ as well.
 * @returns A fresh network with a reset layer counter and η = 0.01. Calling
 * `feedForward()` on it throws, since the hidden layer has no σ.
 */
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

/**
 * Asserts a truthy condition, throwing with `message` otherwise.
 *
 * @param condition The value to check for truthiness.
 * @param message Text to include in the error when the check fails. Defaults
 * to 'Assertion failed'.
 * @returns Nothing; acts as a TypeScript assertion function, so the compiler
 * treats `condition` as truthy afterwards.
 * @throws Error If `condition` is falsy.
 */
export function assert(
  condition: unknown,
  message = 'Assertion failed',
): asserts condition {
  if (!condition) throw new Error(message)
}

/**
 * Asserts deep equality between two values (numbers, strings, nested arrays
 * and plain objects).
 *
 * @param actual The value produced by the code under test.
 * @param expected The value it should equal. Compared structurally, so key
 * order in objects is irrelevant.
 * @param message Optional prefix for the failure message, identifying the test.
 * @returns Nothing.
 * @throws Error If the two values are not deeply equal. Arrays are compared
 * element-wise and must have equal length; objects are compared by own
 * enumerable keys and must have the same number of them.
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

/**
 * Asserts that `actual` is within `eps` of `expected`, for float comparisons.
 *
 * Rejects non-finite `actual` outright, because NaN silently satisfies a naive
 * `Math.abs(actual - expected) > eps` check: the difference is NaN, and every
 * comparison against NaN is false. Without this guard a test handed a
 * non-numeric value would pass vacuously.
 *
 * @param actual The computed value.
 * @param expected The reference value.
 * @param eps Maximum allowed absolute difference; defaults to 1e-9, which is
 * tight enough for the hand-computed expectations used in these tests.
 * @param message Optional prefix for the failure message.
 * @returns Nothing.
 * @throws Error If `actual` is not a finite number, or if the absolute
 * difference exceeds `eps`.
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

/**
 * Asserts element-wise closeness of two numeric arrays.
 *
 * @param actual The computed vector.
 * @param expected The reference vector; must have the same length as `actual`.
 * @param eps Maximum allowed absolute difference per element; defaults to 1e-9.
 * @param message Optional prefix for the failure message.
 * @returns Nothing.
 * @throws Error If the lengths differ, or if any element is further than `eps`
 * from its counterpart. The length check happens first, so a mismatch never
 * compares entries past the shorter array.
 */
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
 *
 * @param fn The function expected to throw. It is invoked once.
 * @param match A substring the message must contain, or a RegExp tested against
 * it. Matching on the message keeps assertions loose enough to survive
 * rewording while still pinning the failure cause.
 * @returns Nothing.
 * @throws Error If `fn` completes without throwing, or if the message it threw
 * does not match. A non-Error throw is stringified before matching.
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

/**
 * Structural equality used by {@link assertEquals}: identical primitives, or
 * arrays/objects compared element-wise by own keys.
 *
 * @param actual The value produced by the code under test.
 * @param expected The value it should equal.
 * @returns Whether the two values are structurally equal. `Object.is` handles
 * the primitives first, so `NaN` equals `NaN` and `+0` does not equal `-0`.
 */
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

/**
 * Renders a value for an assertion failure message.
 *
 * @param value The value to render; any type.
 * @returns Its JSON representation, or `String(value)` when JSON.stringify
 * returns undefined (e.g. for `undefined` or a function), so the message never
 * reads "undefined" without explanation.
 */
function fmt(value: unknown): string {
  return JSON.stringify(value) ?? String(value)
}
