import { Network } from '../network.ts'
import { Layer, ActivationFunctionType } from '../layer.ts'

/**
 * Resets the static `Layer.id` counter, so layer indices (𝓁) are deterministic
 * from one test to the next. Mutates shared state and must be called before
 * building a network whose indices a test depends on.
 *
 * @returns Nothing.
 */
export function resetLayerCounter(): void {
  Layer.id = 0
}

/**
 * The standard MNIST-sized network (784 -> 16 -> 16 -> 10), with random weights
 * and tanh-free activations, resetting the layer counter first.
 *
 * @returns A freshly constructed {@link Network} with the standard topology.
 * @throws {Error} Propagated from the {@link Network} constructor if the layer
 * sizes fail its validation (not expected for this fixed topology).
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
 * A small 2 -> 2 -> 2 network, built without the constructor's MNIST size
 * validation so tests can hand-verify the math on concrete numbers. The layer
 * counter is reset first.
 *
 * @param σHidden Activation for the hidden layer, or `undefined` for none.
 * Defaults to `'sigmoid'`.
 * @param σOutput Activation for the output layer, or `undefined` for none.
 * Defaults to `'sigmoid'`.
 * @returns A freshly constructed 2 -> 2 -> 2 {@link Network} bypassing size checks.
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
 * A 2 -> 2 -> 2 network whose hidden layer has no σ, for exercising the
 * "activation function required" error paths. The layer counter is reset first.
 *
 * @param σOutput Activation for the output layer, or `undefined` for none.
 * Defaults to `'sigmoid'`.
 * @returns A {@link Network} with an activation-less hidden layer.
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
 * Asserts a truthy condition. Narrows the type of `condition` for the caller.
 *
 * @param condition The value to test; any falsy value fails the assertion.
 * @param message The error message used on failure. Defaults to
 * `'Assertion failed'`.
 * @returns Nothing.
 * @throws {Error} If `condition` is falsy, with `message` as its text.
 */
export function assert(
  condition: unknown,
  message = 'Assertion failed',
): asserts condition {
  if (!condition) throw new Error(message)
}

/**
 * Asserts deep equality, comparing arrays element-wise and objects by key
 * (see {@link deepEqual}). Primitives are compared with `Object.is`.
 *
 * @param actual The value produced by the code under test.
 * @param expected The value it should equal.
 * @param message Optional prefix for the failure message. Defaults to `''`.
 * @returns Nothing.
 * @throws {Error} If the two values are not deeply equal, formatted as
 * `"<message>: Expected <actual> to equal <expected>"`.
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
 * A non-finite `actual` is rejected, since it would pass the comparison
 * vacuously.
 *
 * @param actual The computed value.
 * @param expected The target value.
 * @param eps Maximum allowed absolute difference. Defaults to `1e-9`.
 * @param message Optional prefix for the failure message. Defaults to `''`.
 * @returns Nothing.
 * @throws {Error} If `actual` is not finite, or if `|actual - expected|` exceeds
 * `eps`.
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
 * Asserts that two numeric arrays are element-wise within `eps` via
 * {@link assertClose}.
 *
 * @param actual The computed array.
 * @param expected The target array.
 * @param eps Maximum allowed absolute difference per element. Defaults to `1e-9`.
 * @param message Optional prefix for failure messages. Defaults to `''`.
 * @returns Nothing.
 * @throws {Error} If the arrays differ in length, or if any element pair fails
 * {@link assertClose}.
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
 * Asserts that `fn` throws, and that the thrown message contains `match` (a
 * substring or RegExp).
 *
 * @param fn The zero-argument function expected to throw when called.
 * @param match A substring or RegExp the error message must satisfy.
 * @returns Nothing.
 * @throws {Error} If `fn` returns without throwing, or if it throws a message
 * that does not match `match`.
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
 * Structural equality behind {@link assertEquals}, recursing into arrays and
 * objects. Primitives (and `NaN`, `-0`) are compared with `Object.is`.
 *
 * @param actual The first value to compare.
 * @param expected The second value to compare.
 * @returns `true` if the two values are structurally equal, otherwise `false`.
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
 * Renders a value for a failure message, falling back to `String(value)`.
 *
 * @param value The value to render.
 * @returns `JSON.stringify(value)` when it produces a string (including
 * `"null"`), otherwise `String(value)` — covering values `JSON.stringify`
 * yields `undefined` for, such as functions and `undefined`.
 */
function fmt(value: unknown): string {
  return JSON.stringify(value) ?? String(value)
}
