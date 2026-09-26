import { NumericMatrix, NumericVector } from './types.ts'

/**
 * Re-maps a `value` from the range [lower, higher] to the range
 * [projectedLower, projectedUpper]. Optionally clamps the output to the target
 * range when `withinBounds` is true (any input outside the source range is
 * snapped to the nearest endpoint). Used e.g. to turn a pixel brightness in
 * [0, 1] into an index into a Unicode character gradient.
 */
export const map = (
  value: number,
  lower: number,
  higher: number,
  projectedLower: number,
  projectedUpper: number,
  withinBounds = false,
) => {
  // Clamp input when withinBounds is set. Handles inverted source ranges too.
  if (withinBounds) {
    if (lower <= higher) {
      if (value <= lower) return projectedLower
      else if (value >= higher) return projectedUpper
    } else if (value >= lower) {
      return projectedLower
    } else if (value <= higher) {
      return projectedUpper
    }
  }

  // Linear interpolation: map(value) = ((value - lower) / span) * span + offset
  return (
    ((value - lower) / (higher - lower)) * (projectedUpper - projectedLower) +
    projectedLower
  )
}

/**
 * Executes `fn` once for each index in [0, count). The array-free equivalent
 * of `for (let i = 0; i < count; i++)`, when you don't need a return value.
 */
export const timesForEach = (count: number, fn: (i: number) => void) => {
  for (let i = 0; i < count; i++) fn(i)
}

/**
 * Builds an array of length `count` where slot i is `fn(i)`.
 * The array-building counterpart to {@link timesForEach}.
 */
export const timesMap = <T>(count: number, fn: (index: number) => T): T[] => {
  const results: T[] = []

  for (let i = 0; i < count; i++) results[i] = fn(i)

  return results
}

/**
 * Reduces over the indices [0, count) via `fn`, optionally starting from
 * `initialAcc`. Without an initial value, index 0 becomes the accumulator
 * (which is why the loop starts at 1 in that case).
 */
export const timesReduce = <T>(
  count: number,
  fn: (acc: T, item: number) => T,
  initialAcc?: T,
): T => {
  const startIndex = initialAcc === undefined ? 1 : 0
  let acc = initialAcc ?? (0 as T)

  for (let i = startIndex; i < count; i++) acc = fn(acc, i)

  return acc
}

/** Iterates a collection from its last element down to the first. */
export const reversedForEach = <T>(
  collection: T[],
  fn: (item: T, index: number) => void,
) => {
  for (let i = collection.length - 1; i >= 0; i--) fn(collection[i], i)
}

/**
 * Recursive type to create nested arrays based on the length of the dimensions tuple.
 * [number, number] -> T[][]
 */
type NestedArray<T, D extends number[]> = D extends [
  number,
  ...infer Rest extends number[],
]
  ? NestedArray<T, Rest>[]
  : T

type ArrayAsObject<D extends number[]> = { [K in keyof D]: number }

/**
 * Generates an N-dimensional array.
 * @param dimensions A tuple or array defining the size of each dimension.
 * @param callback A function receiving all current indices and returning the value.
 */
export const timesMapN = <T, D extends number[]>(
  dimensions: [...D],
  callback: (...indexes: ArrayAsObject<D>) => T,
): NestedArray<T, D> => {
  // Internal helper to track accumulated indices through recursion
  const accumulateIndices = (
    currentDimensions: number[],
    currentIndexes: number[],
    // deno-lint-ignore no-explicit-any
  ): any => {
    const [firstDimension, ...remainingDimensions] = currentDimensions
    const augmentedIndexes = (i: number) =>
      [...currentIndexes, i] as ArrayAsObject<D>

    return timesMap(firstDimension, i =>
      remainingDimensions.length === 0
        ? callback(...augmentedIndexes(i))
        : accumulateIndices(remainingDimensions, [...currentIndexes, i]),
    )
  }

  // Handle empty dimensions case
  if (dimensions.length === 0) return [] as NestedArray<T, D>

  return accumulateIndices(dimensions, [])
}

export const timesForEachN = <T, D extends number[]>(
  dimensions: [...D],
  callback: (...indexes: ArrayAsObject<D>) => T,
): void => {
  // Internal helper to track accumulated indices through recursion
  const accumulateIndices = (
    currentDimensions: number[],
    currentIndexes: number[],
  ): void => {
    const [firstDimension, ...remainingDimensions] = currentDimensions
    const augmentedIndexes = (i: number) =>
      [...currentIndexes, i] as ArrayAsObject<D>

    timesForEach(firstDimension, i => {
      if (remainingDimensions.length === 0) {
        callback(...augmentedIndexes(i))
      } else {
        accumulateIndices(remainingDimensions, [...currentIndexes, i])
      }
    })
  }

  // Handle empty dimensions case
  if (dimensions.length === 0) return

  accumulateIndices(dimensions, [])
}

export function assertIsNotUndefined<T>(
  val: T | undefined | null,
): asserts val is NonNullable<T> | null {
  if (val === undefined)
    throw new TypeError(
      `Expected value not to be undefined, but received ${val}`,
    )
}

export function assertIsNotNull<T>(
  val: T | undefined | null,
): asserts val is NonNullable<T> | undefined {
  if (val === null)
    throw new TypeError(`Expected value not to be null, but received ${val}`)
}

export function assertIsNotUndefinedOrNull<T>(
  val: T | undefined | null,
): asserts val is NonNullable<T> {
  if (val === undefined || val === null)
    throw new TypeError(
      `Expected value not to be undefined or null, but received ${val}`,
    )
}

/**
 * Converts an Nx1 column vector (a list of numbers) into an Nx1 matrix, so it
 * can participate in matrix operations.
 * @example [1, 2, 3] -> [[1], [2], [3]]
 */
export const toMatrix = (vector: number[]): number[][] =>
  vector.map(value => [value])

/**
 * Inverse of {@link toMatrix}: extracts the single column from an Nx1 matrix.
 * @example [[1], [2], [3]] -> [1, 2, 3]
 */
export const fromMatrix = (matrix: number[][]): number[] =>
  matrix.map(row => row[0])

/** Element-wise addition of two matrices with identical dimensions. */
export const addMatrices = (
  left: NumericMatrix,
  right: NumericMatrix,
): NumericMatrix => {
  const rowsLeft = left.length
  const colsLeft = left[0]?.length ?? 0
  const rowsRight = right.length
  const colsRight = right[0]?.length ?? 0

  if (colsLeft !== colsRight)
    throw new Error(
      `Number of columns from left matrix (${colsLeft}) must match number of columns from right one (${colsRight})`,
    )

  if (rowsLeft !== rowsRight)
    throw new Error(
      `Number of rows from left matrix (${rowsLeft}) must match number of rows from right one (${rowsRight})`,
    )

  return timesMapN(
    [rowsLeft, colsLeft],
    (row, col) => left[row][col] + right[row][col],
  )
}

/**
 * Standard matrix product (rows of `left` dotted with columns of `right`).
 * Requires left.cols === right.rows. Used to compute z(𝓁) = w(𝓁)·a(𝓁-1).
 */
export const multiplyMatrices = (
  left: NumericMatrix,
  right: NumericMatrix,
): NumericMatrix => {
  const colsLeft = left[0].length
  const colsRight = right[0].length
  const rowsLeft = left.length
  const rowsRight = right.length

  if (colsLeft !== rowsRight)
    throw new Error(
      `Number of columns from left matrix (${colsLeft}) must match number of rows from right one (${rowsRight})`,
    )

  const result: NumericMatrix = []

  for (let row = 0; row < rowsLeft; row++) {
    result[row] = []

    for (let col = 0; col < colsRight; col++) {
      result[row][col] = 0

      // colsLeft = rowsRight
      for (let i = 0; i < colsLeft; i++)
        result[row][col] += left[row][i] * right[i][col]
    }
  }

  return result
}

export const transposeMatrix = (matrix: NumericMatrix): NumericMatrix => {
  const cols = matrix[0].length
  const rows = matrix.length

  const result: NumericMatrix = []

  timesForEachN([rows, cols], (row, col) => {
    result[col] ??= []
    result[col][row] = matrix[row][col]
  })

  return result
}

export const multiplyMatrixByScalar = (
  matrix: NumericMatrix,
  scalar: number,
): NumericMatrix => {
  const rowsLeft = matrix.length
  const colsLeft = matrix[0]?.length ?? 0

  return timesMapN(
    [rowsLeft, colsLeft],
    (row, col) => matrix[row][col] * scalar,
  )
}

export const hadamardProduct = (
  left: NumericVector,
  right: NumericVector,
): NumericVector => {
  if (left.length !== right.length)
    throw new Error(
      `Size of left vector (${left.length}) must match size of right one (${right.length})`,
    )

  return left.map((leftValue, i) => leftValue * right[i])
}

export const addVectors = (
  left: NumericVector,
  right: NumericVector,
): NumericVector => {
  if (left.length !== right.length)
    throw new Error(
      `Size of left vector (${left.length}) must match size of right one (${right.length})`,
    )

  return left.map((leftValue, i) => leftValue + right[i])
}

export const multiplyVectorByScalar = (
  vector: NumericVector,
  scalar: number,
): NumericVector => {
  return vector.map(value => value * scalar)
}

/** Returns a random number in [min, max). Defaults to [0, 1). */
export const random = (min = 0, max = 1) => Math.random() * (max - min) + min

/**
 * Fisher-Yates (Knuth) shuffle: randomizes array in-place in O(n).
 * Used to ensure each epoch sees training data in a different order.
 */
export const shuffle = (array: number[]): void => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))

    ;[array[i], array[j]] = [array[j], array[i]]
  }
}
