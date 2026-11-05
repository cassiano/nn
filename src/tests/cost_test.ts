///////////////////////
// AI-generated code //
///////////////////////

import {
  costDerivative,
  costValue,
  crossEntropy,
  crossEntropyDerivative,
  meanSquaredError,
  meanSquaredErrorDerivative,
} from '../cost.ts'
import { assertArrayClose, assertClose } from './test_helpers.ts'

Deno.test('meanSquaredError / sums the squared differences', () => {
  assertClose(
    meanSquaredError([1, 0], [0.75, 0.1]),
    (1 - 0.75) ** 2 + (0 - 0.1) ** 2,
  )
  assertClose(meanSquaredError([1, 0], [1, 0]), 0)
})

Deno.test('meanSquaredErrorDerivative / follows 2(a - y)', () => {
  assertArrayClose(
    meanSquaredErrorDerivative([1, 0], [0.75, 0.1]),
    [2 * (0.75 - 1), 2 * (0.1 - 0)],
    1e-12,
  )
})

Deno.test('crossEntropy / sums -y log(a)', () => {
  assertClose(crossEntropy([0, 1], [0.2, 0.8]), -Math.log(0.8))
  assertClose(crossEntropy([1, 0], [1, 0]), 0)
})

Deno.test('crossEntropy / supports soft targets', () => {
  const target = [0.25, 0.75]
  const output = [0.4, 0.6]

  assertClose(
    crossEntropy(target, output),
    -(0.25 * Math.log(0.4) + 0.75 * Math.log(0.6)),
  )
})

Deno.test('crossEntropyDerivative / follows -y / a', () => {
  // A zero target contributes nothing, even when its output is also zero.
  assertArrayClose(
    crossEntropyDerivative([0, 1], [0, 0.8]),
    [0, -1 / 0.8],
    1e-12,
  )
})

Deno.test('cost functions / derivatives agree with finite differences', () => {
  const target = [0.2, 0.8]
  const output = [0.35, 0.65]
  const eps = 1e-6

  const centralDifference = (cost: (output: number[]) => number, i: number) => {
    const plus = [...output]
    const minus = [...output]
    plus[i] += eps
    minus[i] -= eps

    return (cost(plus) - cost(minus)) / (2 * eps)
  }

  const mse = meanSquaredErrorDerivative(target, output)
  const crossEntropyGradient = crossEntropyDerivative(target, output)

  for (let i = 0; i < output.length; i++) {
    assertClose(
      mse[i],
      centralDifference(values => meanSquaredError(target, values), i),
      1e-6,
    )
    assertClose(
      crossEntropyGradient[i],
      centralDifference(values => crossEntropy(target, values), i),
      1e-6,
    )
  }
})

Deno.test('costValue / selects the configured cost', () => {
  const target = [0, 1]
  const output = [0.2, 0.8]

  assertClose(costValue('mse', target, output), meanSquaredError(target, output))
  assertClose(
    costValue('cross-entropy', target, output),
    crossEntropy(target, output),
  )
})

Deno.test('costDerivative / selects the configured derivative', () => {
  const target = [0, 1]
  const output = [0.2, 0.8]

  assertArrayClose(
    costDerivative('mse', target, output),
    meanSquaredErrorDerivative(target, output),
    1e-12,
  )
  assertArrayClose(
    costDerivative('cross-entropy', target, output),
    crossEntropyDerivative(target, output),
    1e-12,
  )
})
