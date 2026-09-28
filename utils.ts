import { NumericMatrix, NumericVector } from './types.ts'

/**
 * Re-maps a `value` from the range [lower, higher] to the range
 * [projectedLower, projectedUpper]. Optionally clamps the output to the target
 * range when `withinBounds` is true (any input outside the source range is
 * snapped to the nearest endpoint). Used e.g. to turn a pixel brightness in
 * [0, 1] into an index into a Unicode character gradient.
 *
 * @param value The number to re-map.
 * @param lower Inclusive lower bound of the source range.
 * @param higher Inclusive upper bound of the source range. May be below
 * `lower`, which describes a descending source range.
 * @param projectedLower Value that `lower` maps to.
 * @param projectedUpper Value that `higher` maps to. May be below
 * `projectedLower`, which describes a descending target range.
 * @param withinBounds When true (default false), values outside the source
 * range snap to the nearest projected endpoint instead of extrapolating. Also
 * avoids a division by zero when `lower` equals `higher`, since the guard
 * returns before the interpolation runs.
 * @returns The re-mapped value, or a projected endpoint when clamped.
 * @returns NaN or ±Infinity when `lower` equals `higher` and `withinBounds` is
 * false, since the interpolation divides by a zero-width range.
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
 *
 * @param count How many times to invoke `fn`. Values below 1 mean `fn` is
 * never called.
 * @param fn Called with each index in ascending order, from 0 to `count - 1`.
 * @returns Nothing.
 */
export const timesForEach = (count: number, fn: (i: number) => void) => {
  for (let i = 0; i < count; i++) fn(i)
}

/**
 * Builds an array of length `count` where slot i is `fn(i)`.
 * The array-building counterpart to {@link timesForEach}.
 *
 * @param count Length of the array to build. Values below 1 produce `[]`.
 * @param fn Produces the value at each index, called with indices 0 to
 * `count - 1` in ascending order. Returning `undefined` leaves a hole in the
 * array, so the result is not guaranteed to be dense.
 * @returns An array of `count` entries, or `[]` when `count` is 0.
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
 *
 * @param count Number of indices to visit, from 0 to `count - 1`.
 * @param fn Combines the running accumulator with the current index and
 * returns the next accumulator.
 * @param initialAcc Starting value. When omitted, the accumulator starts at
 * `0` and the first index passed to `fn` is 1, so index 0 is never folded in;
 * `count` of 1 therefore returns the seed untouched.
 * @returns The final accumulated value, or `initialAcc` (or 0) when `count` is
 * 0 and `fn` was never called.
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

/**
 * Iterates a collection from its last element down to the first.
 *
 * @param collection The array to walk. A copy is not made, so the callback
 * must not mutate it.
 * @param fn Called with each item and its index, starting from the last index
 * and counting down to 0.
 * @returns Nothing. Use {@link timesMap} if you need to collect results.
 */
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

/**
 * Maps a dimensions tuple to the tuple of positional arguments its callback
 * receives, e.g. `[number, number]` -> `(row: number, col: number)`.
 */
type ArrayAsObject<D extends number[]> = { [K in keyof D]: number }

/**
 * Generates an N-dimensional array, recursing through the dimension sizes and
 * calling `callback` once per combination of indices.
 *
 * @param dimensions A tuple or array defining the size of each dimension, in
 * nesting order (outermost first). An empty array yields `[]` at runtime.
 * @param callback A function receiving all current indices and returning the
 * value stored at that position, e.g. `(row, col)` for `[2, 3]`.
 * @returns A nested array shaped by `dimensions`, with `callback`'s results at
 * the leaves.
 * @example timesMapN([2, 2], (row, col) => row * 2 + col)
 * // => [[0, 1], [2, 3]]
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

/**
 * Visits every index combination of an N-dimensional space exactly once,
 * recursing through the dimension sizes, without building an array.
 *
 * The side-effecting counterpart to {@link timesMapN}.
 *
 * @param dimensions A tuple or array defining the size of each dimension, in
 * nesting order (outermost first). An empty array means `callback` is never
 * called.
 * @param callback A function receiving all current indices, e.g. `(row, col)`
 * for `[2, 3]`. Its return value is ignored.
 * @returns Nothing.
 */
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

/**
 * Narrows a value known to be possibly `undefined` to one that is not, throwing
 * otherwise. Acts as a TypeScript assertion function, so the compiler treats
 * `val` as `NonNullable<T> | null` afterwards. Note `null` is allowed through:
 * use {@link assertIsNotNull} to reject it.
 *
 * @param val The value to check.
 * @returns Nothing; returns normally only when `val` is not `undefined`.
 * @throws TypeError If `val` is `undefined`.
 */
export function assertIsNotUndefined<T>(
  val: T | undefined | null,
): asserts val is NonNullable<T> | null {
  if (val === undefined)
    throw new TypeError(
      `Expected value not to be undefined, but received ${val}`,
    )
}

/**
 * Narrows a value known to be possibly `null` to one that is not, throwing
 * otherwise. Acts as a TypeScript assertion function, so the compiler treats
 * `val` as `NonNullable<T> | undefined` afterwards. Note `undefined` is allowed
 * through: use {@link assertIsNotUndefined} to reject it.
 *
 * @param val The value to check.
 * @returns Nothing; returns normally only when `val` is not `null`.
 * @throws TypeError If `val` is `null`.
 */
export function assertIsNotNull<T>(
  val: T | undefined | null,
): asserts val is NonNullable<T> | undefined {
  if (val === null)
    throw new TypeError(`Expected value not to be null, but received ${val}`)
}

/**
 * Narrows a value known to be possibly `undefined` or `null` to one that is
 * neither, throwing otherwise. Acts as a TypeScript assertion function, so the
 * compiler treats `val` as `NonNullable<T>` afterwards.
 *
 * @param val The value to check.
 * @returns Nothing; returns normally only when `val` is neither `undefined`
 * nor `null`.
 * @throws TypeError If `val` is `undefined` or `null`.
 */
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
 *
 * @param vector The values to wrap, one per row.
 * @returns A matrix with one single-element row per entry. An empty vector
 * yields `[]`.
 * @example toMatrix([1, 2, 3]) // => [[1], [2], [3]]
 */
export const toMatrix = (vector: number[]): number[][] =>
  vector.map(value => [value])

/**
 * Inverse of {@link toMatrix}: extracts the single column from an Nx1 matrix.
 *
 * @param matrix An Nx1 matrix, i.e. one single-entry row per value.
 * @returns The first entry of each row. An empty matrix yields `[]`, and rows
 * longer than one entry have their extra columns dropped.
 * @example fromMatrix([[1], [2], [3]]) // => [1, 2, 3]
 */
export const fromMatrix = (matrix: number[][]): number[] =>
  matrix.map(row => row[0])

/**
 * Element-wise addition of two matrices with identical dimensions.
 *
 * @param left The matrix whose values appear in the result.
 * @param right The matrix added onto `left`; must have the same shape.
 * @returns A new matrix of the same shape holding `left[i][j] + right[i][j]`.
 * Two empty matrices yield `[]`.
 * @throws If the column counts differ, or if the column counts match but the
 * row counts do not.
 */
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
 *
 * @param left The left factor, of shape [rows][cols].
 * @param right The right factor, of shape [cols][colsRight]; its row count must
 * equal `left`'s column count.
 * @returns A new matrix of shape [left.rows][right.cols] holding the row-by-
 * column dot products. Operands are not mutated.
 * @throws If `left`'s column count differs from `right`'s row count, or if
 * either operand is empty.
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

/**
 * Swaps a matrix's rows and columns, reflecting it along its main diagonal.
 * Used to broadcast a weight matrix across activations during backpropagation.
 *
 * @param matrix The matrix to transpose, of shape [rows][cols].
 * @returns A new matrix of shape [cols][rows] holding `matrix[i][j]` at
 * `[j][i]`. Transposing twice returns the original; operands are not mutated.
 * @throws If `matrix` is empty, since its column count is read from row 0.
 * @example
 * transposeMatrix([[1, 2, 3], [4, 5, 6]]) // => [[1, 4], [2, 5], [3, 6]]
 */
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

/**
 * Multiplies every entry of a matrix by a single scalar, used to scale a
 * gradient by the learning rate.
 *
 * @param matrix The matrix to scale, of shape [rows][cols].
 * @param scalar The factor applied to every entry; may be negative.
 * @returns A new matrix of the same shape. An empty matrix yields `[]`.
 */
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

/**
 * Element-wise (Hadamard) product of two equally sized vectors.
 *
 * @param left The left factor.
 * @param right The right factor; must have the same length as `left`.
 * @returns A new vector holding `left[i] * right[i]`.
 * @throws If the two vectors have different lengths.
 */
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

/**
 * Element-wise addition of two equally sized vectors, used to step a layer's
 * biases against their gradient.
 *
 * @param left The vector whose values appear in the result.
 * @param right The vector added onto `left`; must have the same length.
 * @returns A new vector holding `left[i] + right[i]`. Operands are not
 * mutated, and two empty vectors yield `[]`.
 * @throws If the two vectors have different lengths.
 */
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

/**
 * Multiplies every entry of a vector by a single scalar, used to scale a
 * gradient by the learning rate.
 *
 * @param vector The vector to scale.
 * @param scalar The factor applied to every entry; may be negative.
 * @returns A new vector of the same length. An empty vector yields `[]`.
 */
export const multiplyVectorByScalar = (
  vector: NumericVector,
  scalar: number,
): NumericVector => {
  return vector.map(value => value * scalar)
}

/**
 * Returns a random number in [min, max). Defaults to [0, 1).
 *
 * @param min Inclusive lower bound of the range; defaults to 0.
 * @param max Exclusive upper bound of the range; defaults to 1.
 * @returns A pseudo-random float in [min, max), uniform over that interval.
 */
export const random = (min = 0, max = 1) => Math.random() * (max - min) + min

/**
 * Fisher-Yates (Knuth) shuffle: randomizes array in-place in O(n).
 * Used to ensure each epoch sees training data in a different order.
 *
 * Mutates the array by swapping, so pass a copy if the original must survive.
 * Uniform over permutations, and never mutates `Math.random`.
 *
 * @param array The array to shuffle in place. Arrays of length 0 or 1 are left
 * untouched, as there is only one permutation.
 * @returns Nothing.
 */
export const shuffle = (array: number[]): void => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))

    ;[array[i], array[j]] = [array[j], array[i]]
  }
}

export const formatWithDecimalPlaces = (value: number, decimalPlaces: number) =>
  Math.round(value * 100 * 10 ** decimalPlaces) / 10 ** decimalPlaces
