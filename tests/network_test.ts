///////////////////////
// AI-generated code //
///////////////////////

import { sigmoid } from '../activation.ts'
import {
  makeStandardNetwork,
  makeTinyNetwork,
  makeTinyNetworkWithoutActivation,
  resetLayerCounter,
  sampleGradient,
} from './test_helpers.ts'
import {
  assert,
  assertEquals,
  assertThrows,
  assertClose,
  assertArrayClose,
} from './test_helpers.ts'
import { Network } from '../network.ts'
import { Layer } from '../layer.ts'
import { Gradient } from '../types.ts'
import { divideGradientByScalar } from '../utils.ts'
import { BATCH_SIZE } from '../constants.ts'
import {
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
  MNIST_OUTPUT_SIZE,
} from '../mnist_loader.ts'

/**
 * Builds a 2 -> 2 -> 2 network with fixed weights, so the gradient can be
 * checked against central differences on hand-picked numbers.
 */
function makeSeededTinyNetwork(): Network {
  const net = makeTinyNetwork()
  const [input, hidden, output] = net.layers

  input.a = [0.5, 0.9]
  hidden.w = [
    [0.7, -0.2],
    [0.1, 0.4],
  ]
  hidden.b = [0.1, -0.3]
  output.w = [
    [1, 0.5],
    [-0.5, 2],
  ]
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
        [
          { name: 'in', size: 1 },
          { name: 'out', size: 10 },
        ],
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
        [
          { name: 'in', size: 784 },
          { name: 'out', size: 3 },
        ],
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
  assertEquals(
    net.parameterCount,
    net.layers.reduce((s, l) => s + l.parameterCount, 0),
  )
})

Deno.test(
  'Network / constructor initializes all parameters randomly in [-1, 1)',
  () => {
    const net = makeStandardNetwork()
    for (const layer of net.layers.slice(1)) {
      for (const row of layer.w)
        for (const value of row) assert(value >= -1 && value < 1)
      for (const value of layer.b) assert(value >= -1 && value < 1)
    }
  },
)

Deno.test(
  'Network / loadSample stores inputs and one-hot-encodes the label',
  () => {
    const net = makeStandardNetwork()
    const inputs = Array.from({ length: 784 }, (_, i) => i / 784)
    net.loadSample(inputs, 4)
    assertEquals(net.inputLayer.a, inputs)
    assertEquals(net.y, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
  },
)

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
  hidden.w = [
    [0.7, -0.2],
    [0.1, 0.4],
  ]
  hidden.b = [0.1, -0.3]
  output.w = [
    [1, 0.5],
    [-0.5, 2],
  ]
  output.b = [0.2, -0.1]

  net.feedForward()

  // Hidden layer: z1[0] = 0.7*0.5 + (-0.2)*0.9 + 0.1, z1[1] = 0.1*0.5 + 0.4*0.9 - 0.3
  const z1 = [0.7 * 0.5 + -0.2 * 0.9 + 0.1, 0.1 * 0.5 + 0.4 * 0.9 + -0.3]
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

Deno.test(
  'Network / output layer gradient matches numerical differentiation',
  () => {
    const net = makeSeededTinyNetwork()
    const [, , output] = net.layers

    // sampleGradient() orders entries output layer first, so index 0 is 𝐋.
    const { w, b } = sampleGradient(net)[0]

    assertClose(w[0][0], numericPartial(net, output, 0, 0), 1e-4)
    assertClose(w[1][1], numericPartial(net, output, 1, 1), 1e-4)
    assertClose(b[0], numericPartial(net, output, 0, null), 1e-4)
    assertClose(b[1], numericPartial(net, output, 1, null), 1e-4)
  },
)

Deno.test(
  'Network / inner layer gradient matches numerical differentiation',
  () => {
    const net = makeSeededTinyNetwork()
    const [, hidden] = net.layers

    // The hidden layer's entry sits after the output layer's, at index 1.
    const { 𝓁, w, b } = sampleGradient(net)[1]
    assertEquals(𝓁, 1)

    // The error reaching the hidden layer is propagated from the output layer
    // through w(2)ᵀ · δ(2), not read straight off y, so its partials must still
    // agree with central differences. This is the regression guard for the
    // placeholder rule that indexed y by neuron and produced NaN.
    assertClose(w[0][0], numericPartial(net, hidden, 0, 0), 1e-4)
    assertClose(w[1][1], numericPartial(net, hidden, 1, 1), 1e-4)
    assertClose(b[0], numericPartial(net, hidden, 0, null), 1e-4)
    assertClose(b[1], numericPartial(net, hidden, 1, null), 1e-4)
  },
)

Deno.test(
  'Network / a sample gradient covers every trainable layer once',
  () => {
    const net = makeTinyNetwork()
    const [input] = net.layers
    input.a = [0.5, 0.9]
    net.y = [1, 0]
    net.feedForward()

    const gradient = sampleGradient(net)

    // One entry per layer that owns parameters; the input layer has none.
    assertEquals(gradient.length, net.layers.length - 1)
    // Ordered output layer first, then each inner layer descending to 1.
    assertEquals(
      gradient.map(entry => entry.𝓁),
      [2, 1],
    )

    // Each entry mirrors the shape of the parameters it belongs to.
    for (const { 𝓁, w, b } of gradient) {
      const layer = net.layers[𝓁]
      assertEquals(w.length, layer.w.length)
      assertEquals(w[0].length, layer.w[0].length)
      assertEquals(b.length, layer.b.length)
    }
  },
)

Deno.test(
  'Network / gradient holds one partial per trainable parameter',
  () => {
    // Each layer contributes ∂C/∂w for every weight and ∂C/∂b for every bias,
    // so the flattened partial count must equal the network's parameterCount.
    const net = makeStandardNetwork()
    net.loadSample(new Array(784).fill(0.5), 3)
    net.feedForward()

    const gradient = sampleGradient(net)
    const partialCount = gradient.reduce(
      (sum, { w, b }) => sum + w.length * w[0].length + b.length,
      0,
    )

    assertEquals(partialCount, net.parameterCount)
    assertEquals(net.parameterCount, 13002)
  },
)

Deno.test('Network / gradient partials are finite and defined', () => {
  // The error term is propagated backwards through w(𝓁+1)ᵀ · δ(𝓁+1), so no
  // layer may read past the end of y. This is the regression guard for the
  // placeholder rule that produced NaN across the 16-neuron hidden layers.
  const net = makeStandardNetwork()
  net.loadSample(new Array(784).fill(0.5), 3)
  net.feedForward()

  for (const { w, b } of sampleGradient(net)) {
    for (const partial of b) assert(Number.isFinite(partial))
    for (const row of w)
      for (const partial of row) assert(Number.isFinite(partial))
  }
})

Deno.test(
  'Network / accumulateGradient throws when a layer has no activation',
  () => {
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
      () => withoutOutput.accumulateGradient(withoutOutput.createEmptyGradient()),
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
      () => withoutHidden.accumulateGradient(withoutHidden.createEmptyGradient()),
      'Expected activation function to be defined',
    )
  },
)

Deno.test('Network / createEmptyGradient is zeroed, output layer first', () => {
  const net = makeStandardNetwork()

  const gradient = net.createEmptyGradient()

  // One entry per trainable layer, ordered output first, as the walk produces.
  assertEquals(gradient.map(entry => entry.𝓁), [3, 2, 1])

  // Shapes mirror the parameters, and every entry starts at zero.
  const shapes: [number, number][] = [
    [10, 16],
    [16, 16],
    [16, 784],
  ]
  gradient.forEach((entry, i) => {
    const [rows, cols] = shapes[i]
    assertEquals(entry.w.length, rows)
    assertEquals(entry.w[0].length, cols)
    assertEquals(entry.b.length, rows)
    assertEquals(
      [...entry.w.flat(), ...entry.b].every(value => value === 0),
      true,
    )
  })
})

Deno.test('Network / accumulateGradient sums repeated samples', () => {
  // Called twice without retraining in between, the buffer holds twice the
  // single-sample gradient — that is what lets a batch share one buffer.
  const net = makeStandardNetwork()
  net.loadSample(new Array(784).fill(0.5), 2)
  net.feedForward()

  const once = sampleGradient(net)

  const sum = net.createEmptyGradient()
  net.accumulateGradient(sum)
  net.accumulateGradient(sum)

  assertArrayClose(
    sum.flatMap(entry => [...entry.w.flat(), ...entry.b]),
    once.flatMap(entry => [...entry.w.flat(), ...entry.b]).map(value => 2 * value),
    1e-12,
  )
})

Deno.test(
  'Network / accumulateGradient throws when the target is missing a layer',
  () => {
    const net = makeStandardNetwork()
    net.loadSample(new Array(784).fill(0.5), 1)
    net.feedForward()

    // An empty target has no entry for the output layer the walk starts from.
    assertThrows(
      () => net.accumulateGradient([]),
      'not found in target gradient',
    )
  },
)

Deno.test(
  'Network / accumulateGradient + divideGradientByScalar gives the batch mean',
  () => {
    // The batched training step: sum each sample into one buffer, then scale by
    // 1/n. Compared against an independent, obviously-correct mean over the same
    // batch.
    const net = makeStandardNetwork()
    const samples = [0, 1, 2, 3, 4]

    const perSample: Gradient[] = []
    const sum = net.createEmptyGradient()

    for (const label of samples) {
      net.loadSample(new Array(784).fill(label / samples.length), label)
      net.feedForward()
      perSample.push(sampleGradient(net))
      net.accumulateGradient(sum)
    }

    const mean = divideGradientByScalar(sum, samples.length)

    for (const { 𝓁, w, b } of mean) {
      const entries = perSample.map(g => g.find(e => e.𝓁 === 𝓁)!)

      for (let r = 0; r < w.length; r++)
        for (let c = 0; c < w[r].length; c++)
          assertClose(
            w[r][c],
            entries.reduce((total, e) => total + e.w[r][c], 0) / samples.length,
            1e-12,
          )

      for (let r = 0; r < b.length; r++)
        assertClose(
          b[r],
          entries.reduce((total, e) => total + e.b[r], 0) / samples.length,
          1e-12,
        )
    }
  },
)

Deno.test(
  'Network / backPropagate steps the weights against the gradient',
  () => {
    // Exact arithmetic: the update is w -= η·BATCH_SIZE·∂C/∂w, with the BATCH_SIZE
    // factor cancelling the 1/BATCH_SIZE that divideGradientByScalar divides by.
    const net = makeTinyNetwork()
    const [input, hidden, output] = net.layers
    input.a = [0.5, 0.9]
    hidden.w = [
      [0.7, -0.2],
      [0.1, 0.4],
    ]
    hidden.b = [0.1, -0.3]
    output.w = [
      [1, 0.5],
      [-0.5, 2],
    ]
    output.b = [0.2, -0.1]

    const gradient: Gradient = [
      {
        𝓁: 2,
        w: [
          [1, 2],
          [3, 4],
        ],
        b: [5, 6],
      },
      {
        𝓁: 1,
        w: [
          [7, 8],
          [9, 10],
        ],
        b: [11, 12],
      },
    ]

    net.backPropagate(gradient)

    const rate = net.η * BATCH_SIZE
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        assertClose(
          output.w[r][c],
          [
            [1, 0.5],
            [-0.5, 2],
          ][r][c] -
            rate * gradient[0].w[r][c],
          1e-12,
        )
        assertClose(
          hidden.w[r][c],
          [
            [0.7, -0.2],
            [0.1, 0.4],
          ][r][c] -
            rate * gradient[1].w[r][c],
          1e-12,
        )
      }
    }
    assertClose(output.b[0], 0.2 - rate * 5, 1e-12)
    assertClose(output.b[1], -0.1 - rate * 6, 1e-12)
    assertClose(hidden.b[0], 0.1 - rate * 11, 1e-12)
    assertClose(hidden.b[1], -0.3 - rate * 12, 1e-12)
  },
)

Deno.test(
  'Network / backPropagate applies entries by their 𝓁, not their position',
  () => {
    // The gradient walk emits the output layer first, but the update is addressed
    // by 𝓁, so a reordered gradient must still land on the right layers.
    const net = makeTinyNetwork()
    const [input, hidden, output] = net.layers
    input.a = [0.5, 0.9]
    hidden.w = [
      [0, 0],
      [0, 0],
    ]
    hidden.b = [0, 0]
    output.w = [
      [0, 0],
      [0, 0],
    ]
    output.b = [0, 0]

    const outputFirst: Gradient = [
      {
        𝓁: 2,
        w: [
          [1, 0],
          [0, 0],
        ],
        b: [0, 0],
      },
      {
        𝓁: 1,
        w: [
          [0, 0],
          [0, 1],
        ],
        b: [0, 0],
      },
    ]

    net.backPropagate([...outputFirst].reverse())

    // Only (0,0) of the output layer and (1,1) of the hidden layer move; the
    // entry emitted second in the reversed list must not touch the output layer.
    assertClose(output.w[0][0], -net.η * BATCH_SIZE, 1e-12)
    assertClose(output.w[1][0], 0, 1e-12)
    assertClose(hidden.w[1][1], -net.η * BATCH_SIZE, 1e-12)
    assertClose(hidden.w[0][0], 0, 1e-12)
  },
)

Deno.test('Network / backPropagate throws for a 𝓁 that is not a layer', () => {
  const net = makeTinyNetwork()
  net.layers[0].a = [1, 1]
  net.y = [1, 0]
  net.feedForward()

  const gradient: Gradient = sampleGradient(net).map(entry => ({
    ...entry,
    𝓁: entry.𝓁 + 10, // no such layer in a 2 -> 2 -> 2 network
  }))

  assertThrows(() => net.backPropagate(gradient), 'Layer 12 not found')
})

Deno.test(
  'Network / backPropagate throws when a gradient shape does not fit',
  () => {
    const net = makeTinyNetwork()
    net.layers[0].a = [1, 1]
    net.y = [1, 0]
    net.feedForward()

    const gradient = sampleGradient(net)
    // A 3x3 weight gradient against a 2x2 weight matrix. The column counts are
    // compared first, so that is what the error names.
    gradient[0].w = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]

    assertThrows(
      () => net.backPropagate(gradient),
      'must match number of columns',
    )
  },
)

Deno.test(
  'Network / backPropagate reduces the cost of the current sample',
  () => {
    // One real gradient step on a fixed, hand-computable network. η is scaled by
    // 1/BATCH_SIZE so the effective step stays small, since a step of η·BATCH_SIZE
    // would overshoot this toy network.
    const net = makeSeededTinyNetwork()
    net.η = 0.01 / BATCH_SIZE

    const costBefore = net.cost
    net.backPropagate(sampleGradient(net))
    net.feedForward()

    assert(net.cost < costBefore, `expected cost to fall from ${costBefore}`)
  },
)

Deno.test('Network / a mini-batch step lowers the batch total cost', () => {
  // The whole learning path in one go: per-sample gradients, their mean, a
  // single update, then every sample of the same batch re-scored against the
  // new weights. The invariant a batch update gives is on the *sum* of the
  // batch's costs, not on each sample individually — one mean direction cannot
  // lower every sample's cost at once, so only the total is claimed here.
  // The seed makes the random initialization deterministic: the unseeded
  // network overshoots (the batch cost rises) for a small fraction of initial
  // weights, which made this test flaky.
  const net = makeStandardNetwork(622)
  net.η = 0.01 / BATCH_SIZE

  const samples = [0, 1, 2, 3].map(i => ({
    inputs: new Array(784).fill(i / 3),
    label: i,
  }))

  const summedGradient = net.createEmptyGradient()

  for (const { inputs, label } of samples) {
    net.loadSample(inputs, label)
    net.feedForward()
    net.accumulateGradient(summedGradient)
  }

  const totalCostBefore = samples.reduce((total, { inputs, label }) => {
    net.loadSample(inputs, label)
    net.feedForward()
    return total + net.cost
  }, 0)

  net.backPropagate(divideGradientByScalar(summedGradient, samples.length))

  const totalCostAfter = samples.reduce((total, { inputs, label }) => {
    net.loadSample(inputs, label)
    net.feedForward()
    return total + net.cost
  }, 0)

  assert(
    totalCostAfter < totalCostBefore,
    `expected the batch cost to fall from ${totalCostBefore}, got ${totalCostAfter}`,
  )
})

Deno.test(
  'Network / predictedDigit returns the largest activation index',
  () => {
    const net = makeStandardNetwork()
    net.outputLayer.a = [0, 0, 0, 0.9, 0, 0, 0.1, 0, 0, 0]

    assertEquals(net.predictedDigit(), 3)
  },
)

Deno.test(
  'Network / predictedDigit breaks ties towards the first index',
  () => {
    // findIndex returns the first match, so a flat output layer predicts 0 rather
    // than an arbitrary digit.
    const net = makeStandardNetwork()
    net.outputLayer.a = new Array(MNIST_OUTPUT_SIZE).fill(0.1)

    assertEquals(net.predictedDigit(), 0)
  },
)

Deno.test('Network / previousLayer and nextLayer walk the stack', () => {
  const net = makeStandardNetwork()
  const [input, hidden1, hidden2, output] = net.layers

  assertEquals(net.previousLayer(1), input)
  assertEquals(net.nextLayer(1), hidden2)
  assertEquals(net.previousLayer(2), hidden1)
  assertEquals(net.nextLayer(2), output)
})

Deno.test('Network / previousLayer rejects the input layer', () => {
  const net = makeStandardNetwork()

  assertThrows(
    () => net.previousLayer(0),
    'Input layer does not have a previous one',
  )
})

Deno.test('Network / nextLayer rejects the output layer', () => {
  const net = makeStandardNetwork()

  assertThrows(
    () => net.nextLayer(net.𝐋),
    'Output layer does not have a next one',
  )
})
