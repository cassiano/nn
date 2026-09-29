import {
  map,
  timesForEach,
  timesMap,
  timesReduce,
  reversedForEach,
  timesMapN,
  timesForEachN,
  assertIsNotUndefined,
  assertIsNotNull,
  assertIsNotUndefinedOrNull,
  createMatrix,
  createVector,
  toMatrix,
  fromMatrix,
  addMatrices,
  addVectors,
  multiplyMatrices,
  transposeMatrix,
  multiplyMatrixByScalar,
  multiplyVectorByScalar,
  divideMatrixByScalar,
  divideVectorByScalar,
  hadamardProduct,
  random,
  shuffle,
  gradientAsVector,
  formatPercentageWithDecimalPlaces,
} from '../utils.ts'
import { Gradient } from '../types.ts'
import {
  makeTinyNetwork,
  assert,
  assertEquals,
  assertThrows,
  assertClose,
  assertArrayClose,
} from './test_helpers.ts'

Deno.test('map / performs linear interpolation', () => {
  assertEquals(map(5, 0, 10, 0, 100), 50)
  assertEquals(map(0, 0, 10, 0, 100), 0)
  assertEquals(map(10, 0, 10, 0, 100), 100)
  assertEquals(map(2.5, 0, 10, 0, 100), 25)
  assertEquals(map(3, 1, 5, 100, 200), 150)
})

Deno.test('map / handles inverted source ranges', () => {
  assertEquals(map(5, 10, 0, 0, 100), 50)
})

Deno.test('map / clamps when withinBounds is true', () => {
  assertEquals(map(-1, 0, 10, 0, 100, true), 0)
  assertEquals(map(0, 0, 10, 0, 100, true), 0)
  assertEquals(map(11, 0, 10, 0, 100, true), 100)
  assertEquals(map(10, 0, 10, 0, 100, true), 100)
  assertEquals(map(5, 0, 10, 0, 100, true), 50)
})

Deno.test('map / does not clamp when withinBounds is false', () => {
  assertEquals(map(20, 0, 10, 0, 100), 200)
  assertEquals(map(-20, 0, 10, 0, 100), -200)
})

Deno.test('map / supports descending target ranges', () => {
  assertEquals(map(0, 0, 10, 100, 0), 100)
  assertEquals(map(5, 0, 10, 100, 0), 50)
  assertEquals(map(10, 0, 10, 100, 0), 0)
})

Deno.test('map / supports negative source and target ranges', () => {
  assertEquals(map(-5, -10, 0, 0, 100), 50)
  assertEquals(map(0, -10, 0, -1, 1), 1)
})

Deno.test('map / a degenerate source range still clamps', () => {
  // lower === higher avoids the division by zero whenever withinBounds snaps
  // the value, since the guard returns before the interpolation runs.
  assertEquals(map(4, 5, 5, 0, 100, true), 0)
  assertEquals(map(6, 5, 5, 0, 100, true), 100)
})

Deno.test('map / clamps inverted ranges correctly', () => {
  assertEquals(map(11, 10, 0, 0, 100, true), 0) // value >= lower(10) -> projectedLower
  assertEquals(map(-1, 10, 0, 0, 100, true), 100) // value <= higher(0) -> projectedUpper
})

Deno.test('timesForEach / calls the callback with each index in order', () => {
  const seen: number[] = []
  timesForEach(4, i => seen.push(i))
  assertEquals(seen, [0, 1, 2, 3])
})

Deno.test('timesForEach / zero iterations for a zero count', () => {
  let calls = 0
  timesForEach(0, () => calls++)
  assertEquals(calls, 0)
})

Deno.test('timesMap / builds an array from the callback results', () => {
  assertEquals(
    timesMap(3, i => i * 2),
    [0, 2, 4],
  )
  assertEquals(
    timesMap(0, () => 1),
    [],
  )
  assertEquals(
    timesMap(3, () => 'x'),
    ['x', 'x', 'x'],
  )
})

Deno.test('timesReduce / accumulates from the provided initial value', () => {
  assertEquals(
    timesReduce(5, (acc, i) => acc + i, 100),
    110,
  ) // 100 + 0 + 1 + 2 + 3 + 4
})

Deno.test(
  'timesReduce / without initial value seeds the accumulator with 0',
  () => {
    // Implementation detail: starts at index 1 with an accumulator of 0,
    // so count 5 sums {1, 2, 3, 4} = 10.
    assertEquals(
      timesReduce<number>(5, (acc, i) => acc + i),
      10,
    )
  },
)

Deno.test('timesReduce / a zero count returns the initial accumulator', () => {
  assertEquals(
    timesReduce(0, (acc, i) => acc + i, 100),
    100,
  )
  assertEquals(
    timesReduce<number>(0, (acc, i) => acc + i),
    0,
  )
})

Deno.test('timesReduce / count 1 without an initial never calls fn', () => {
  let calls = 0
  const result = timesReduce<number>(1, (acc, i) => {
    calls++
    return acc + i
  })
  assertEquals(result, 0)
  assertEquals(calls, 0)
})

Deno.test(
  'reversedForEach / visits items from last to first with indices',
  () => {
    const seen: (string | number)[] = []
    reversedForEach(['a', 'b', 'c'], (item, index) => seen.push(index, item))
    assertEquals(seen, [2, 'c', 1, 'b', 0, 'a'])
  },
)

Deno.test('reversedForEach / an empty collection never calls fn', () => {
  let calls = 0
  reversedForEach([], () => calls++)
  assertEquals(calls, 0)
})

Deno.test('timesMapN / generates nested arrays with all indices', () => {
  const grid = timesMapN([2, 3], (row, col) => row * 3 + col)
  assertEquals(grid, [
    [0, 1, 2],
    [3, 4, 5],
  ])
})

Deno.test('timesMapN / supports three dimensions', () => {
  const cube = timesMapN([2, 2, 2], (a, b, c) => a + b + c)
  assertEquals(cube, [
    [
      [0, 1],
      [1, 2],
    ],
    [
      [1, 2],
      [2, 3],
    ],
  ])
})

Deno.test(
  'timesMapN / empty dimensions produce an empty array at runtime',
  () => {
    assertEquals(
      timesMapN([] as [], () => 0),
      [],
    )
  },
)

Deno.test('timesForEachN / visits every combination exactly once', () => {
  const visits: number[] = []
  timesForEachN([2, 2], (row, col) => visits.push(row * 2 + col))
  assertEquals(visits, [0, 1, 2, 3])
})

Deno.test('timesForEachN / empty dimensions never call the callback', () => {
  let calls = 0
  timesForEachN([] as [], () => calls++)
  assertEquals(calls, 0)
})

Deno.test('timesForEachN / a zero-sized dimension produces no visits', () => {
  let calls = 0
  timesForEachN([0, 3], () => calls++)
  assertEquals(calls, 0)
})

Deno.test('assertIsNotUndefined / rejects undefined but allows null', () => {
  assertThrows(() => assertIsNotUndefined(undefined), 'not to be undefined')
  assertIsNotUndefined(null) // null is allowed; acts as a type guard
  assertIsNotUndefined(3)
})

Deno.test('assertIsNotNull / rejects null but allows undefined', () => {
  assertThrows(() => assertIsNotNull(null), 'not to be null')
  assertIsNotNull(undefined) // undefined is allowed
  assertIsNotNull(3)
})

Deno.test('assertIsNotUndefinedOrNull / rejects both empty values', () => {
  assertThrows(
    () => assertIsNotUndefinedOrNull(undefined),
    'not to be undefined or null',
  )
  assertThrows(
    () => assertIsNotUndefinedOrNull(null),
    'not to be undefined or null',
  )
  assertIsNotUndefinedOrNull(3)
})

Deno.test('toMatrix / flattens a vector into a column matrix', () => {
  assertEquals(toMatrix([1, 2, 3]), [[1], [2], [3]])
})

Deno.test('toMatrix / an empty vector yields an empty matrix', () => {
  assertEquals(toMatrix([]), [])
})

Deno.test('fromMatrix / is the inverse of toMatrix', () => {
  assertEquals(fromMatrix([[1], [2], [3]]), [1, 2, 3])
  assertEquals(fromMatrix(toMatrix([4, 5, 6])), [4, 5, 6])
})

Deno.test('fromMatrix / an empty matrix yields an empty vector', () => {
  assertEquals(fromMatrix([]), [])
})

Deno.test('addMatrices / sums element-wise', () => {
  assertEquals(
    addMatrices(
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ),
    [
      [6, 8],
      [10, 12],
    ],
  )
})

Deno.test('addMatrices / throws on dimension mismatch', () => {
  assertThrows(() => addMatrices([[1]], [[1, 2]]), 'must match')
  assertThrows(() => addMatrices([[1], [2]], [[1, 2]]), 'must match')
})

Deno.test('addMatrices / distinguishes column and row mismatches', () => {
  // Differing column counts trip the first guard...
  assertThrows(
    () => addMatrices([[1]], [[1, 2]]),
    'Number of columns from left matrix (1) must match',
  )
  // ...while equal columns with differing rows trip the second.
  assertThrows(
    () => addMatrices([[1], [2], [3]], [[1], [2]]),
    'Number of rows from left matrix (3) must match',
  )
})

Deno.test('addMatrices / handles empty matrices', () => {
  assertEquals(addMatrices([], []), [])
  assertEquals(addMatrices([[]], [[]]), [[]])
})

Deno.test('addMatrices / handles negative values', () => {
  assertEquals(addMatrices([[-1, 2]], [[3, -4]]), [[2, -2]])
})

Deno.test('multiplyMatrices / computes the standard matrix product', () => {
  assertEquals(
    multiplyMatrices(
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ),
    [
      [19, 22],
      [43, 50],
    ],
  )
})

Deno.test('multiplyMatrices / supports rectangular matrices', () => {
  const left = [
    [1, 2, 3],
    [4, 5, 6],
  ]
  const right = [
    [7, 8],
    [9, 10],
    [11, 12],
  ]
  assertEquals(multiplyMatrices(left, right), [
    [58, 64],
    [139, 154],
  ])
})

Deno.test('multiplyMatrices / 1x1 matrices multiply like scalars', () => {
  assertEquals(multiplyMatrices([[3]], [[4]]), [[12]])
})

Deno.test('multiplyMatrices / the identity changes nothing', () => {
  const left = [
    [1, 2, 3],
    [4, 5, 6],
  ]
  // Left: a 3x3 identity on the right, 2x3 == 2x3. Right: a 2x2 identity on
  // the left, 2 == 2 rows. Both must reproduce `left`.
  assertEquals(
    multiplyMatrices(left, [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]),
    left,
  )
  assertEquals(
    multiplyMatrices(
      [
        [1, 0],
        [0, 1],
      ],
      left,
    ),
    left,
  )
})

Deno.test('multiplyMatrices / handles negative values', () => {
  assertEquals(multiplyMatrices([[-1, 2]], [[3], [-4]]), [[-11]])
})

Deno.test('multiplyMatrices / throws when inner dimensions differ', () => {
  assertThrows(() => multiplyMatrices([[1, 2]], [[1, 2]]), 'must match')
})

Deno.test(
  'multiplyMatrices / multiplying 2 vectors (as matrixes) can be done in any order, as long as their shapes properly align',
  () => {
    const vectorA = [2, 3, 4]
    const vectorB = [10, 20, 30, 40]

    assertEquals(
      multiplyMatrices(toMatrix(vectorA), transposeMatrix(toMatrix(vectorB))),
      [
        [20, 40, 60, 80],
        [30, 60, 90, 120],
        [40, 80, 120, 160],
      ],
    )
    assertEquals(
      transposeMatrix(
        multiplyMatrices(toMatrix(vectorB), transposeMatrix(toMatrix(vectorA))),
      ),
      [
        [20, 40, 60, 80],
        [30, 60, 90, 120],
        [40, 80, 120, 160],
      ],
    )
  },
)

Deno.test('transposeMatrix / swaps rows and columns', () => {
  assertEquals(
    transposeMatrix([
      [1, 2],
      [3, 4],
    ]),
    [
      [1, 3],
      [2, 4],
    ],
  )
})

Deno.test('transposeMatrix / handles rectangular matrices', () => {
  assertEquals(
    transposeMatrix([
      [1, 2, 3],
      [4, 5, 6],
    ]),
    [
      [1, 4],
      [2, 5],
      [3, 6],
    ],
  )
  assertEquals(
    transposeMatrix([
      [1, 4],
      [2, 5],
      [3, 6],
    ]),
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
  )
})

Deno.test('transposeMatrix / 1x1 matrices are unchanged', () => {
  assertEquals(transposeMatrix([[7]]), [[7]])
})

Deno.test('transposeMatrix / is its own inverse', () => {
  const original = [
    [1, 2, 3],
    [4, 5, 6],
  ]
  assertEquals(transposeMatrix(transposeMatrix(original)), original)
})

Deno.test('transposeMatrix / swaps the shape of the result', () => {
  const transposed = transposeMatrix([
    [1, 2, 3],
    [4, 5, 6],
  ])
  assertEquals(transposed.length, 3)
  assertEquals(transposed[0].length, 2)
})

Deno.test('transposeMatrix / preserves every element exactly once', () => {
  const original = [
    [1, 2, 3],
    [4, 5, 6],
  ]
  // Transposing reorders elements, so compare as a sorted multiset.
  const flattenSorted = (m: number[][]) => m.flat().sort((a, b) => a - b)
  assertEquals(
    flattenSorted(transposeMatrix(original)),
    flattenSorted(original),
  )
})

Deno.test('hadamardProduct / multiplies element-wise', () => {
  assertEquals(hadamardProduct([1, 2, 3], [4, 5, 6]), [4, 10, 18])
})

Deno.test('hadamardProduct / zeroes out where either factor is zero', () => {
  assertEquals(hadamardProduct([0, 2, 0], [4, 0, 6]), [0, 0, 0])
})

Deno.test('hadamardProduct / works with single-element vectors', () => {
  assertEquals(hadamardProduct([3], [4]), [12])
})

Deno.test('hadamardProduct / empty vectors yield an empty vector', () => {
  assertEquals(hadamardProduct([], []), [])
})

Deno.test('hadamardProduct / handles negative values', () => {
  assertEquals(hadamardProduct([-1, 2], [3, -4]), [-3, -8])
})

Deno.test('hadamardProduct / throws when vector sizes differ', () => {
  assertThrows(() => hadamardProduct([1, 2], [1, 2, 3]), 'must match')
})

Deno.test('hadamardProduct / is commutative', () => {
  assertEquals(
    hadamardProduct([1, 2, 3], [4, 5, 6]),
    hadamardProduct([4, 5, 6], [1, 2, 3]),
  )
})

Deno.test('random / stays within the default [0, 1) range', () => {
  for (let i = 0; i < 1000; i++) {
    const r = random()
    assert(r >= 0 && r < 1)
  }
})

Deno.test('random / respects a custom range', () => {
  for (let i = 0; i < 1000; i++) {
    const r = random(-1, 1)
    assert(r >= -1 && r < 1)
  }
})

Deno.test(
  'shuffle / permutes deterministically under controlled randomness',
  () => {
    const original = Math.random
    Math.random = () => 0
    try {
      const array = [1, 2, 3]
      shuffle(array)
      // Fisher-Yates with Math.random() === 0 always swaps i with 0:
      // [1,2,3] -> swap(2,0) -> [3,2,1] -> swap(1,0) -> [2,3,1]
      assertEquals(array, [2, 3, 1])
    } finally {
      Math.random = original
    }
  },
)

Deno.test('shuffle / preserves the multiset of elements', () => {
  const original = [1, 2, 3, 4, 5, 6, 7, 8]
  const copy = [...original]
  shuffle(copy)
  assertEquals(
    [...copy].sort((a, b) => a - b),
    original,
  )
})

Deno.test('shuffle / leaves arrays of zero or one element untouched', () => {
  const empty: number[] = []
  shuffle(empty)
  assertEquals(empty, [])

  const single = [42]
  shuffle(single)
  assertEquals(single, [42])
})

Deno.test('shuffle / mutates the array in place and returns nothing', () => {
  const array = [1, 2, 3]
  assertEquals(shuffle(array), undefined)
  assertEquals(array.length, 3)
})

Deno.test('createMatrix / fills every cell with the given value', () => {
  assertEquals(createMatrix(2, 3, 7), [
    [7, 7, 7],
    [7, 7, 7],
  ])
  assertEquals(createMatrix(1, 1, -0.5), [[-0.5]])
})

Deno.test('createMatrix / calls the function once per cell, in order', () => {
  // Row-major order, one invocation per cell, which is what a random weight
  // initializer relies on for an independent value per weight.
  const calls: number[] = []
  const matrix = createMatrix(2, 2, () => {
    calls.push(calls.length)
    return calls.length
  })

  assertEquals(matrix, [
    [1, 2],
    [3, 4],
  ])
  assertEquals(calls, [0, 1, 2, 3])
})

Deno.test('createMatrix / zero dimensions produce an empty matrix', () => {
  assertEquals(createMatrix(0, 5, 1), [])
  // Zero columns still yields the requested number of rows, each of them empty:
  // the row count is filled in before the column count is consulted.
  assertEquals(createMatrix(2, 0, 1), [[], []])
})

Deno.test('createVector / fills every entry with the given value', () => {
  assertEquals(createVector(3, 0), [0, 0, 0])
  assertEquals(createVector(1, 2.5), [2.5])
})

Deno.test('createVector / calls the function once per entry', () => {
  let counter = 0
  assertEquals(
    createVector(3, () => ++counter),
    [1, 2, 3],
  )
})

Deno.test('createVector / size 0 produces an empty vector', () => {
  assertEquals(createVector(0, 1), [])
})

Deno.test(
  'createVector / matches the size of a same-shaped createMatrix column',
  () => {
    const vector = createVector(3, 0)
    const column = createMatrix(3, 1, 0).map(([value]) => value)

    assertEquals(vector, column)
  },
)

Deno.test('multiplyMatrixByScalar / scales every entry', () => {
  assertEquals(
    multiplyMatrixByScalar(
      [
        [1, 2],
        [3, 4],
      ],
      2,
    ),
    [
      [2, 4],
      [6, 8],
    ],
  )
  assertEquals(multiplyMatrixByScalar([[1.5, -2]], 0.5), [[0.75, -1]])
})

Deno.test('multiplyMatrixByScalar / accepts a negative factor', () => {
  // backPropagate relies on this: the gradient is scaled by -η to move the
  // weights against it.
  assertEquals(multiplyMatrixByScalar([[1, -2]], -0.1), [[-0.1, 0.2]])
})

Deno.test('multiplyMatrixByScalar / an empty matrix stays empty', () => {
  assertEquals(multiplyMatrixByScalar([], 3), [])
})

Deno.test('multiplyVectorByScalar / scales every entry', () => {
  assertArrayClose(multiplyVectorByScalar([1, 2, 3], 2), [2, 4, 6])
  assertArrayClose(multiplyVectorByScalar([1, -1], 0.25), [0.25, -0.25])
})

Deno.test('multiplyVectorByScalar / an empty vector stays empty', () => {
  assertEquals(multiplyVectorByScalar([], 3), [])
})

Deno.test('divideMatrixByScalar / divides every entry', () => {
  assertArrayClose(
    divideMatrixByScalar(
      [
        [2, 4],
        [6, 8],
      ],
      2,
    ).flat(),
    [1, 2, 3, 4],
  )
  assertArrayClose(divideMatrixByScalar([[1]], 4).flat(), [0.25])
})

Deno.test('divideMatrixByScalar / undoes multiplyMatrixByScalar', () => {
  const matrix = [
    [0.1, -0.2],
    [0.3, 0.4],
  ]

  assertEquals(
    divideMatrixByScalar(multiplyMatrixByScalar(matrix, 7), 7),
    matrix,
  )
})

Deno.test('divideMatrixByScalar / throws on a zero divisor', () => {
  // Rejected up front: a silent divide would fill the whole gradient with
  // Infinity/NaN and the weights with them on the next step.
  assertThrows(
    () => divideMatrixByScalar([[1]], 0),
    'Cannot divide matrix by 0',
  )
})

Deno.test('divideMatrixByScalar / an empty matrix stays empty', () => {
  assertEquals(divideMatrixByScalar([], 2), [])
})

Deno.test('divideVectorByScalar / divides every entry', () => {
  assertArrayClose(divideVectorByScalar([2, -4], 2), [1, -2])
  assertArrayClose(divideVectorByScalar([1], 4), [0.25])
})

Deno.test('divideVectorByScalar / undoes multiplyVectorByScalar', () => {
  const vector = [0.1, -0.2, 0.3]

  assertArrayClose(
    divideVectorByScalar(multiplyVectorByScalar(vector, 60), 60),
    vector,
    1e-12,
  )
})

Deno.test('divideVectorByScalar / throws on a zero divisor', () => {
  assertThrows(() => divideVectorByScalar([1], 0), 'Cannot divide vector by 0')
})

Deno.test('divideVectorByScalar / an empty vector stays empty', () => {
  assertEquals(divideVectorByScalar([], 2), [])
})

Deno.test('addVectors / sums element-wise', () => {
  assertEquals(addVectors([1, 2, 3], [10, 20, 30]), [11, 22, 33])
})

Deno.test('addVectors / handles negative values', () => {
  assertEquals(addVectors([1, -2], [-1, 2]), [0, 0])
})

Deno.test('addVectors / throws when the sizes differ', () => {
  assertThrows(() => addVectors([1, 2], [1]), 'Size of left vector')
})

Deno.test('addVectors / two empty vectors yield an empty vector', () => {
  assertEquals(addVectors([], []), [])
})

Deno.test('addVectors / does not mutate its operands', () => {
  const left = [1, 2]
  const right = [3, 4]

  addVectors(left, right)

  assertEquals(left, [1, 2])
  assertEquals(right, [3, 4])
})

Deno.test(
  'formatPercentageWithDecimalPlaces / rounds to the requested precision',
  () => {
    // The value is a ratio, so the output is a percentage with the requested
    // number of decimal places.
    assertClose(formatPercentageWithDecimalPlaces(0.985678, 2), 98.57, 1e-9)
    assertClose(formatPercentageWithDecimalPlaces(0.5, 2), 50, 1e-9)
    assertClose(formatPercentageWithDecimalPlaces(1, 2), 100, 1e-9)
  },
)

Deno.test(
  'formatPercentageWithDecimalPlaces / zero decimal places rounds to whole percent',
  () => {
    assertClose(formatPercentageWithDecimalPlaces(0.985678, 0), 99, 1e-9)
  },
)

Deno.test(
  'formatPercentageWithDecimalPlaces / leaves an already exact value alone',
  () => {
    assertClose(formatPercentageWithDecimalPlaces(0.5, 2), 50, 1e-9)
    assertClose(formatPercentageWithDecimalPlaces(0, 2), 0, 1e-9)
  },
)

Deno.test(
  'formatPercentageWithDecimalPlaces / rounds the halfway case up',
  () => {
    // Math.round is the rounding rule: 0.125 -> 12.5 -> 13.
    assertClose(formatPercentageWithDecimalPlaces(0.125, 1), 12.5, 1e-9)
    assertClose(formatPercentageWithDecimalPlaces(0.126, 1), 12.6, 1e-9)
    assertClose(formatPercentageWithDecimalPlaces(0.124, 1), 12.4, 1e-9)
  },
)

Deno.test(
  'gradientAsVector / orders layers by 𝓁 and flattens weights then biases',
  () => {
    // The layout: one contiguous block per layer, ascending 𝓁, each block being
    // the layer's weights in row-major order followed by its biases.
    const gradient: Gradient = [
      { 𝓁: 2, w: [[4], [5]], b: [6] },
      { 𝓁: 1, w: [[1, 2]], b: [3] },
    ]

    assertEquals(gradientAsVector(gradient), [1, 2, 3, 4, 5, 6])
  },
)

Deno.test(
  'gradientAsVector / does not depend on the order the layers arrive in',
  () => {
    // The sort exists for this: the same gradient expressed in any order has to
    // flatten to the same vector, otherwise position in the vector would depend
    // on how the caller happened to assemble the gradient.
    const first: Gradient = [
      { 𝓁: 1, w: [[1, 2]], b: [3] },
      { 𝓁: 2, w: [[4], [5]], b: [6] },
    ]
    const reversed: Gradient = [...first].reverse()

    assertEquals(gradientAsVector(reversed), gradientAsVector(first))
    assertEquals(
      gradientAsVector([first[1], first[0]]),
      gradientAsVector(first),
    )
  },
)

Deno.test(
  'gradientAsVector / flattens a weight matrix in row-major order',
  () => {
    // A 2x2 matrix reads left to right, top to bottom — the same order the
    // matrix is indexed with, so w[i][j] keeps a predictable slot.
    const gradient: Gradient = [
      {
        𝓁: 1,
        w: [
          [1, 2],
          [3, 4],
        ],
        b: [5, 6],
      },
    ]

    assertEquals(gradientAsVector(gradient), [1, 2, 3, 4, 5, 6])
  },
)

Deno.test('gradientAsVector / keeps every layer block contiguous', () => {
  // A layer's biases follow all of that layer's weights, not its own row: no
  // value from the next layer may slip in between w and b.
  const gradient: Gradient = [
    {
      𝓁: 1,
      w: [
        [1, 2],
        [3, 4],
      ],
      b: [5, 6],
    },
    { 𝓁: 2, w: [[7], [8], [9]], b: [10] },
  ]

  assertEquals(gradientAsVector(gradient), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
})

Deno.test('gradientAsVector / yields one entry per trainable parameter', () => {
  const gradient: Gradient = [
    {
      𝓁: 1,
      w: [
        [1, 2, 3],
        [4, 5, 6],
      ],
      b: [7, 8],
    }, // 6 weights + 2 biases
    {
      𝓁: 2,
      w: [
        [9, 10],
        [11, 12],
        [13, 14],
      ],
      b: [15, 16, 17],
    },
  ]
  const expectedLength = gradient.reduce(
    (total, layer) => total + layer.w.flat().length + layer.b.length,
    0,
  )

  assertEquals(gradientAsVector(gradient).length, expectedLength)
})

Deno.test('gradientAsVector / leaves the gradient untouched', () => {
  // toSorted copies, so the caller's array keeps both its order and its
  // contents — a flat view must not cost the caller its gradient.
  const gradient: Gradient = [
    { 𝓁: 2, w: [[4], [5]], b: [6] },
    { 𝓁: 1, w: [[1, 2]], b: [3] },
  ]
  const snapshot = JSON.parse(JSON.stringify(gradient))

  // Poking the result must not write through to the gradient, so it is a fresh
  // array rather than a view onto the layers' own arrays.
  const vector = gradientAsVector(gradient)
  vector[0] = 999

  assertEquals(gradient, snapshot)
  assertEquals(
    gradient.map(layer => layer.𝓁),
    [2, 1],
  )
  assertEquals(vector, [999, 2, 3, 4, 5, 6])
})

Deno.test('gradientAsVector / an empty gradient yields an empty vector', () => {
  assertEquals(gradientAsVector([]), [])
})

Deno.test(
  'gradientAsVector / flattens a real gradient in ascending 𝓁 order',
  () => {
    // The real case the sort was written for: calculateGradient walks the layers
    // from the output backwards, so its gradient arrives in the opposite order to
    // the vector this function produces.
    const net = makeTinyNetwork()
    net.loadSample([1, 0], 1)
    net.feedForward()

    const gradient = net.calculateGradient()

    assertEquals(
      gradient.map(layer => layer.𝓁),
      [2, 1],
    )

    const [output, hidden] = gradient
    const expected = [
      ...hidden.w.flat(),
      ...hidden.b,
      ...output.w.flat(),
      ...output.b,
    ]

    assertArrayClose(gradientAsVector(gradient), expected)

    // And the vector spans the network's parameters exactly: two 2x2 weight
    // matrices, four biases, no input-layer slot.
    assertEquals(gradientAsVector(gradient).length, 2 * 2 * 2 + 4)
  },
)
