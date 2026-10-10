import { NumericMatrix, NumericVector, Gradient } from './types.ts'

/**
 * Re-maps a value from one range onto another by linear interpolation, e.g. a
 * pixel brightness in [0, 1] onto an index into a Unicode character gradient.
 *
 * @param value The value to re-map.
 * @param lower Lower bound of the source range.
 * @param higher Upper bound of the source range.
 * @param projectedLower Value `lower` maps to.
 * @param projectedUpper Value `higher` maps to.
 * @param withinBounds Clamps values outside the source range to the nearest
 * projected endpoint instead of extrapolating.
 * @returns The value projected onto the target range.
 */
export const map = (
  value: number,
  lower: number,
  higher: number,
  projectedLower: number,
  projectedUpper: number,
  withinBounds = false,
): number => {
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
 * Runs `fn` once for each index in [0, count), when no return value is needed.
 *
 * @param count How many indices to visit.
 * @param fn Called with each index in turn.
 * @returns Nothing; `fn`'s own return value, if any, is discarded.
 */
export const timesForEach = (count: number, fn: (i: number) => void): void => {
  for (let i = 0; i < count; i++) fn(i)
}

/**
 * Builds an array of `count` entries where slot i is `fn(i)`.
 *
 * @param count How many entries the result holds.
 * @param fn Called with each index in turn to produce that slot's value.
 * @returns An array of `count` entries, in index order.
 */
export const timesMap = <T>(count: number, fn: (index: number) => T): T[] => {
  const results: T[] = []

  for (let i = 0; i < count; i++) results[i] = fn(i)

  return results
}

/**
 * Folds the indices [0, count) into a single value.
 *
 * @param count How many indices to fold.
 * @param fn Called with the accumulator and the next index, in that order.
 * @param initialAcc Seed to start from. Omitting it starts the accumulator at 0
 * with the first index being 1, so index 0 is never folded in.
 * @returns The accumulated value after the last index.
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
 * Walks a collection from its last element to its first.
 *
 * @param collection The collection to traverse, left unmodified.
 * @param fn Called with each element and its index, last element first.
 * @returns Nothing; `fn`'s own return value, if any, is discarded.
 */
export const reversedForEach = <T>(
  collection: T[],
  fn: (item: T, index: number) => void,
): void => {
  for (let i = collection.length - 1; i >= 0; i--) fn(collection[i], i)
}

/** Maps a dimensions tuple to the positional arguments a callback receives. */
type NestedArray<T, D extends number[]> = D extends [
  number,
  ...infer Rest extends number[],
]
  ? NestedArray<T, Rest>[]
  : T

type ArrayAsObject<D extends number[]> = { [K in keyof D]: number }

/**
 * Builds an N-dimensional array by recursing through the dimension sizes,
 * calling `callback` once per combination of indices.
 *
 * @param dimensions Size of the space along each axis, outermost first.
 * @param callback Called with one index per dimension, in the same order.
 * @returns The nested array `callback` built.
 *
 * @example timesMapN([2, 2], (row, col) => row * 2 + col) // => [[0, 1], [2, 3]]
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

  if (dimensions.length === 0) return [] as NestedArray<T, D>

  return accumulateIndices(dimensions, [])
}

/**
 * Visits every index combination of an N-dimensional space exactly once,
 * without building an array. The side-effecting counterpart to
 * {@link timesMapN}.
 *
 * @param dimensions Size of the space along each axis, outermost first.
 * @param callback Called with one index per dimension, in the same order.
 * @returns Nothing; `callback`'s own return value, if any, is discarded.
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

  if (dimensions.length === 0) return

  accumulateIndices(dimensions, [])
}

/**
 * Asserts a value is not `undefined`, so the compiler can narrow it afterwards.
 * Note `null` passes: use {@link assertIsNotNull} to reject it too.
 *
 * @param val The value to check.
 * @throws If `val` is `undefined`.
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
 * Asserts a value is not `null`, so the compiler can narrow it afterwards. Note
 * `undefined` passes: use {@link assertIsNotUndefined} to reject it too.
 *
 * @param val The value to check.
 * @throws If `val` is `null`.
 */
export function assertIsNotNull<T>(
  val: T | undefined | null,
): asserts val is NonNullable<T> | undefined {
  if (val === null)
    throw new TypeError(`Expected value not to be null, but received ${val}`)
}

/**
 * Asserts a value is neither `undefined` nor `null`, so the compiler can narrow it.
 *
 * @param val The value to check.
 * @throws If `val` is `undefined` or `null`.
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
 * Builds a `rows` x `cols` matrix of a repeated value, or of one value per cell
 * from a function. Used to allocate weight matrices and zero-filled sums.
 *
 * @param rows Number of rows in the result.
 * @param cols Number of columns in the result.
 * @param initialValueOrFn A value every cell starts from, or a function called
 * once per cell.
 * @returns A `rows` x `cols` matrix, row-major.
 *
 * @example createMatrix(2, 2, 0) // => [[0, 0], [0, 0]]
 */
export const createMatrix = (
  rows: number,
  cols: number,
  initialValueOrFn: number | (() => number),
): NumericMatrix => {
  const result: NumericMatrix = new Array(rows)

  for (let row = 0; row < rows; row++) {
    const resultRow: NumericVector = (result[row] = new Array(cols))

    for (let col = 0; col < cols; col++) {
      resultRow[col] =
        typeof initialValueOrFn === 'function'
          ? initialValueOrFn()
          : initialValueOrFn
    }
  }

  return result
}

/**
 * Builds a vector of a repeated value, or of one value per entry from a
 * function. The vector counterpart of {@link createMatrix}.
 *
 * @param size Length of the result.
 * @param initialValueOrFn A value every entry starts from, or a function called
 * once per entry.
 * @returns A vector of `size` entries.
 *
 * @example createVector(3, 0) // => [0, 0, 0]
 */
export const createVector = (
  size: number,
  initialValueOrFn: number | (() => number),
): NumericVector => {
  const result: NumericVector = new Array(size)

  for (let i = 0; i < size; i++) {
    result[i] =
      typeof initialValueOrFn === 'function'
        ? initialValueOrFn()
        : initialValueOrFn
  }

  return result
}

/**
 * Wraps a vector as a single-column matrix, so it can join matrix operations.
 *
 * @param vector The values to wrap, one per row.
 * @returns A one-column matrix holding the same values.
 */
export const toMatrix = (vector: NumericVector): NumericMatrix =>
  vector.map(value => [value])

/**
 * Unpacks a single-column matrix back into a vector.
 *
 * @param matrix The one-column matrix to read.
 * @returns One value per row, in order.
 */
export const toVector = (matrix: NumericMatrix): NumericVector => {
  if (matrix.length === 0) return []

  if (matrix[0]?.length !== 1)
    throw new Error('Expected a single-column matrix')

  return matrix.map(row => row[0])
}

/**
 * Element-wise matrix addition.
 *
 * @param left The first operand.
 * @param right The second operand, of the same shape as `left`.
 * @returns A new matrix holding `left` + `right` cell by cell.
 * @throws If the two matrices do not have the same shape.
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

  const result: NumericMatrix = new Array(rowsLeft)

  for (let row = 0; row < rowsLeft; row++) {
    const leftRow = left[row]
    const rightRow = right[row]
    const resultRow: NumericVector = (result[row] = new Array(colsLeft))

    for (let col = 0; col < colsLeft; col++)
      resultRow[col] = leftRow[col] + rightRow[col]
  }

  return result
}

/**
 * Standard matrix product: the rows of `left` dotted with the columns of
 * `right`. Used to compute z(𝓁) = w(𝓁)·a(𝓁-1).
 *
 * @param left The first operand, whose rows become the result's rows.
 * @param right The second operand, whose columns match `left`'s width.
 * @returns A new matrix of `left`'s rows by `right`'s columns.
 * @throws If the operands are not inner-dimension compatible.
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

  const result: NumericMatrix = new Array(rowsLeft)

  // Accumulating into a local keeps the inner loop down to one multiply and one
  // add per weight, instead of repeatedly re-reading nested array properties.
  for (let row = 0; row < rowsLeft; row++) {
    const leftRow = left[row]
    const resultRow: NumericVector = (result[row] = new Array(colsRight))

    for (let col = 0; col < colsRight; col++) {
      let sum = 0

      for (let i = 0; i < colsLeft; i++) sum += leftRow[i] * right[i][col]

      resultRow[col] = sum
    }
  }

  return result
}

/**
 * Swaps a matrix's rows and columns, reflecting it along its main diagonal.
 *
 * @param matrix The matrix to reflect.
 * @returns A new matrix with the rows and columns exchanged.
 */
export const transposeMatrix = (matrix: NumericMatrix): NumericMatrix => {
  const rows = matrix.length
  const cols = matrix[0]?.length ?? 0

  const result: NumericMatrix = new Array(cols)

  // Pre-allocate all resulting rows.
  for (let col = 0; col < cols; col++) result[col] = new Array(rows)

  for (let row = 0; row < rows; row++) {
    const matrixRow = matrix[row]

    for (let col = 0; col < cols; col++) result[col][row] = matrixRow[col]
  }

  return result
}

/**
 * Multiplies every entry of a matrix by a scalar, used to scale a gradient by
 * the learning rate.
 *
 * @param matrix The matrix to scale.
 * @param scalar The factor applied to every entry.
 * @returns A new matrix of the same shape.
 */
export const multiplyMatrixByScalar = (
  matrix: NumericMatrix,
  scalar: number,
): NumericMatrix => {
  const rows = matrix.length
  const cols = matrix[0]?.length ?? 0

  const result: NumericMatrix = new Array(rows)

  for (let row = 0; row < rows; row++) {
    const matrixRow = matrix[row]
    const resultRow: NumericVector = (result[row] = new Array(cols))

    for (let col = 0; col < cols; col++)
      resultRow[col] = matrixRow[col] * scalar
  }

  return result
}

/**
 * Divides every entry of a matrix by a scalar, used to turn a summed batch
 * gradient into its mean.
 *
 * @param matrix The matrix to scale.
 * @param scalar The divisor applied to every entry.
 * @returns A new matrix of the same shape.
 * @throws If `scalar` is 0, rather than filling the matrix with Infinity/NaN.
 */
export const divideMatrixByScalar = (
  matrix: NumericMatrix,
  scalar: number,
): NumericMatrix => {
  if (scalar === 0) throw new Error('Cannot divide matrix by 0')

  return multiplyMatrixByScalar(matrix, 1 / scalar)
}

/**
 * Element-wise (Hadamard) product of two vectors.
 *
 * @param left The first operand.
 * @param right The second operand, of the same length as `left`.
 * @returns A new vector holding `left` * `right` entry by entry.
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
 * Element-wise vector addition, used to step a layer's biases against their
 * gradient.
 *
 * @param left The first operand.
 * @param right The second operand, of the same length as `left`.
 * @returns A new vector holding `left` + `right` entry by entry.
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
 * Multiplies every entry of a vector by a scalar.
 *
 * @param vector The vector to scale.
 * @param scalar The factor applied to every entry.
 * @returns A new vector of the same length.
 */
export const multiplyVectorByScalar = (
  vector: NumericVector,
  scalar: number,
): NumericVector => {
  return vector.map(value => value * scalar)
}

/**
 * Divides every entry of a vector by a scalar.
 *
 * @param vector The vector to scale.
 * @param scalar The divisor applied to every entry.
 * @returns A new vector of the same length.
 * @throws If `scalar` is 0.
 */
export const divideVectorByScalar = (
  vector: NumericVector,
  scalar: number,
): NumericVector => {
  if (scalar === 0) throw new Error('Cannot divide vector by 0')

  return multiplyVectorByScalar(vector, 1 / scalar)
}

/**
 * A random number in [min, max).
 *
 * @param min Lower bound, included.
 * @param max Upper bound, excluded.
 * @returns A number in [min, max).
 */
export const random = (min = 0, max = 1): number =>
  Math.random() * (max - min) + min

/**
 * Fisher-Yates shuffle, permuting the array in place, so each epoch sees the
 * data in a different order.
 *
 * @param array The array to permute, modified in place.
 * @returns Nothing; `array` is reordered in place.
 */
export const shuffle = <T>(array: T[]): void => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random(0, i + 1))

    ;[array[i], array[j]] = [array[j], array[i]]
  }
}

/**
 * Renders a ratio as a percentage rounded to a fixed precision, so accuracies
 * read the same in every log line.
 *
 * @param value The ratio to render, expected in [0, 1].
 * @param decimalPlaces How many digits to keep after the decimal point.
 * @returns The ratio as a percentage, without a `%` sign.
 *
 * @example formatPercentageWithDecimalPlaces(0.985678, 2) // => 98.57
 */
export const formatPercentageWithDecimalPlaces = (
  value: number,
  decimalPlaces: number,
): number => Math.round(value * 100 * 10 ** decimalPlaces) / 10 ** decimalPlaces

/**
 * Flattens every layer of a gradient into a single vector, so all of a
 * network's parameters can be handled as one list.
 *
 * Layers are ordered by 𝓁, each contributing its weights row-major then its
 * biases, so the result does not depend on the order they arrive in.
 *
 * @param gradient The gradient to flatten.
 * @returns Every weight and bias in canonical order.
 *
 * @example
 * gradientAsVector([
 *   { 𝓁: 2, w: [[4], [5]], b: [6] },
 *   { 𝓁: 1, w: [[1, 2]], b: [3] },
 * ])
 * // => [1, 2, 3, 4, 5, 6]
 */
export const gradientAsVector = (gradient: Gradient): NumericVector => {
  const sortedGradient = gradient.toSorted((left, right) => left.𝓁 - right.𝓁)

  return sortedGradient.flatMap(layer => [...layer.w.flat(2), ...layer.b])
}
