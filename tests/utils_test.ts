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
  toMatrix,
  fromMatrix,
  addMatrices,
  multiplyMatrices,
  transposeMatrix,
  hadamardProduct,
  random,
  shuffle,
} from '../utils.ts'
import { assert, assertEquals, assertThrows } from './test_helpers.ts'

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
  assertEquals(timesMap(3, i => i * 2), [0, 2, 4])
  assertEquals(timesMap(0, () => 1), [])
  assertEquals(timesMap(3, () => 'x'), ['x', 'x', 'x'])
})

Deno.test('timesReduce / accumulates from the provided initial value', () => {
  assertEquals(timesReduce(5, (acc, i) => acc + i, 100), 110) // 100 + 0 + 1 + 2 + 3 + 4
})

Deno.test('timesReduce / without initial value seeds the accumulator with 0', () => {
  // Implementation detail: starts at index 1 with an accumulator of 0,
  // so count 5 sums {1, 2, 3, 4} = 10.
  assertEquals(timesReduce<number>(5, (acc, i) => acc + i), 10)
})

Deno.test('timesReduce / a zero count returns the initial accumulator', () => {
  assertEquals(timesReduce(0, (acc, i) => acc + i, 100), 100)
  assertEquals(timesReduce<number>(0, (acc, i) => acc + i), 0)
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

Deno.test('reversedForEach / visits items from last to first with indices', () => {
  const seen: (string | number)[] = []
  reversedForEach(['a', 'b', 'c'], (item, index) => seen.push(index, item))
  assertEquals(seen, [2, 'c', 1, 'b', 0, 'a'])
})

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

Deno.test('timesMapN / empty dimensions produce an empty array at runtime', () => {
  assertEquals(timesMapN([] as [], () => 0), [])
})

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
  assertThrows(() => assertIsNotUndefinedOrNull(null), 'not to be undefined or null')
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
  assertEquals(addMatrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[6, 8], [10, 12]])
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
  assertEquals(multiplyMatrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]])
})

Deno.test('multiplyMatrices / supports rectangular matrices', () => {
  const left = [[1, 2, 3], [4, 5, 6]]
  const right = [[7, 8], [9, 10], [11, 12]]
  assertEquals(multiplyMatrices(left, right), [[58, 64], [139, 154]])
})

Deno.test('multiplyMatrices / 1x1 matrices multiply like scalars', () => {
  assertEquals(multiplyMatrices([[3]], [[4]]), [[12]])
})

Deno.test('multiplyMatrices / the identity changes nothing', () => {
  const left = [[1, 2, 3], [4, 5, 6]]
  // Left: a 3x3 identity on the right, 2x3 == 2x3. Right: a 2x2 identity on
  // the left, 2 == 2 rows. Both must reproduce `left`.
  assertEquals(multiplyMatrices(left, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), left)
  assertEquals(multiplyMatrices([[1, 0], [0, 1]], left), left)
})

Deno.test('multiplyMatrices / handles negative values', () => {
  assertEquals(multiplyMatrices([[-1, 2]], [[3], [-4]]), [[-11]])
})

Deno.test('multiplyMatrices / throws when inner dimensions differ', () => {
  assertThrows(() => multiplyMatrices([[1, 2]], [[1, 2]]), 'must match')
})

Deno.test('transposeMatrix / swaps rows and columns', () => {
  assertEquals(transposeMatrix([[1, 2], [3, 4]]), [[1, 3], [2, 4]])
})

Deno.test('transposeMatrix / handles rectangular matrices', () => {
  assertEquals(
    transposeMatrix([[1, 2, 3], [4, 5, 6]]),
    [[1, 4], [2, 5], [3, 6]],
  )
  assertEquals(
    transposeMatrix([[1, 4], [2, 5], [3, 6]]),
    [[1, 2, 3], [4, 5, 6]],
  )
})

Deno.test('transposeMatrix / 1x1 matrices are unchanged', () => {
  assertEquals(transposeMatrix([[7]]), [[7]])
})

Deno.test('transposeMatrix / is its own inverse', () => {
  const original = [[1, 2, 3], [4, 5, 6]]
  assertEquals(transposeMatrix(transposeMatrix(original)), original)
})

Deno.test('transposeMatrix / swaps the shape of the result', () => {
  const transposed = transposeMatrix([[1, 2, 3], [4, 5, 6]])
  assertEquals(transposed.length, 3)
  assertEquals(transposed[0].length, 2)
})

Deno.test('transposeMatrix / preserves every element exactly once', () => {
  const original = [[1, 2, 3], [4, 5, 6]]
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

Deno.test('shuffle / permutes deterministically under controlled randomness', () => {
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
})

Deno.test('shuffle / preserves the multiset of elements', () => {
  const original = [1, 2, 3, 4, 5, 6, 7, 8]
  const copy = [...original]
  shuffle(copy)
  assertEquals([...copy].sort((a, b) => a - b), original)
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
