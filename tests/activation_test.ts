import {
  relu,
  reluDerivative,
  sigmoid,
  sigmoidDerivative,
  tanh,
  tanhDerivative,
  softmax,
} from '../activation.ts'
import { assert, assertEquals, assertClose, assertArrayClose } from './test_helpers.ts'

Deno.test('relu / passes positives through, zeroes negatives and zero', () => {
  assertEquals(relu(3), 3)
  assertEquals(relu(0), 0)
  assertEquals(relu(-5), 0)
  assertEquals(relu(0.25), 0.25)
})

Deno.test('reluDerivative / is 1 for positive inputs, 0 otherwise', () => {
  assertEquals(reluDerivative(0.5), 1)
  assertEquals(reluDerivative(123), 1)
  assertEquals(reluDerivative(0), 0)
  assertEquals(reluDerivative(-2), 0)
})

Deno.test('sigmoid / fixed points, asymptotes and symmetry', () => {
  assertEquals(sigmoid(0), 0.5)
  // σ(10) ≈ 1 - 4.5e-5, σ(-10) ≈ 4.5e-5
  assertClose(sigmoid(10), 1, 1e-4)
  assertClose(sigmoid(-10), 0, 1e-4)
  // σ(-x) = 1 - σ(x)
  for (const x of [-5, -2, -1, 0.5, 3]) assertClose(sigmoid(-x), 1 - sigmoid(x))
  // The output always lies strictly in (0, 1) — within these inputs, where
  // double precision has not yet saturated the result.
  for (const x of [-30, -10, 0, 10, 30]) {
    const s = sigmoid(x)
    assert(s > 0 && s < 1)
  }
})

Deno.test('sigmoidDerivative / equals σ(x) * (1 - σ(x))', () => {
  for (const x of [-5, -1, 0, 2, 100])
    assertClose(sigmoidDerivative(x), sigmoid(x) * (1 - sigmoid(x)))
})

Deno.test('tanh / zero crossing, range and odd symmetry', () => {
  assertEquals(tanh(0), 0)
  assertClose(tanh(1), Math.tanh(1))
  for (const x of [-5, 0, 3]) assertClose(tanh(-x), -tanh(x))
  for (const x of [-50, 50]) assert(Math.abs(tanh(x)) <= 1) // rounds to ±1 in float
})

Deno.test('tanhDerivative / equals 1 - tanh(x)^2', () => {
  for (const x of [-3, 0, 2]) assertClose(tanhDerivative(x), 1 - tanh(x) ** 2)
})

Deno.test('softmax / produces a valid probability distribution', () => {
  const logits = [1, 2, 3, 4]
  const p = softmax(logits)
  assertClose(p.reduce((sum, v) => sum + v, 0), 1, 1e-9)
  for (const v of p) assert(v > 0)
  // Largest logit maps to the largest probability (softmax is order-preserving).
  assertEquals(p.indexOf(Math.max(...p)), logits.indexOf(Math.max(...logits)))
  // Explicit value check against the textbook formula.
  const denominator = logits.reduce((sum, v) => sum + Math.exp(v), 0)
  assertClose(p[3], Math.exp(logits[3]) / denominator, 1e-12)
})

Deno.test('softmax / is numerically stable for extreme logits', () => {
  const p = softmax([1000, 1001, 999])
  for (const v of p) assert(Number.isFinite(v))
  // Dominant class approaches probability 1 (σ(10) for the gap).
  const dominant = softmax([0, 10])
  assertClose(dominant[1], 1, 1e-4)
  assertClose(dominant[0], 0, 1e-4)
})

Deno.test('softmax / is invariant to a constant offset', () => {
  assertArrayClose(softmax([1, 2, 3]), softmax([101, 102, 103]), 1e-9)
})

Deno.test('softmax / single-element input maps to 1', () => {
  assertArrayClose(softmax([7]), [1], 1e-9)
})

Deno.test('activation derivatives / agree with finite differences', () => {
  const eps = 1e-6
  const centralDifference = (fn: (x: number) => number, x: number) =>
    (fn(x + eps) - fn(x - eps)) / (2 * eps)

  assertClose(centralDifference(relu, 1), reluDerivative(1), 1e-4)
  assertClose(centralDifference(sigmoid, 0.5), sigmoidDerivative(0.5), 1e-4)
  assertClose(centralDifference(tanh, 0.25), tanhDerivative(0.25), 1e-4)
})