import { NumericMatrix, NumericVector, Gradient } from './types.ts'

/**
 * Re-maps a value from one range onto another by linear interpolation, e.g. a
 * pixel brightness in [0, 1] onto an index into a Unicode character gradient.
 *
 * @param withinBounds Clamps values outside the source range to the nearest
 * projected endpoint instead of extrapolating.
 */
export const map = (
  value: number,
  lower: number,
  higher: number,
  projectedLower: number,
  projectedUpper: number,
  withinBounds = false,
) => {
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

/** Runs `fn` once for each index in [0, count), when no return value is needed. */
export const timesForEach = (count: number, fn: (i: number) => void) => {
  for (let i = 0; i < count; i++) fn(i)
}

/** Builds an array of `count` entries where slot i is `fn(i)`. */
export const timesMap = <T>(count: number, fn: (index: number) => T): T[] => {
  const results: T[] = []

  for (let i = 0; i < count; i++) results[i] = fn(i)

  return results
}

/**
 * Folds the indices [0, count) into a single value.
 *
 * @param initialAcc Seed to start from. Omitting it starts the accumulator at 0
 * with the first index being 1, so index 0 is never folded in.
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

/** Walks a collection from its last element to its first. */
export const reversedForEach = <T>(
  collection: T[],
  fn: (item: T, index: number) => void,
) => {
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
 */
export function assertIsNotNull<T>(
  val: T | undefined | null,
): asserts val is NonNullable<T> | undefined {
  if (val === null)
    throw new TypeError(`Expected value not to be null, but received ${val}`)
}

/** Asserts a value is neither `undefined` nor `null`, so the compiler can narrow it. */
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
 * @example createMatrix(2, 2, 0) // => [[0, 0], [0, 0]]
 */
export const createMatrix = (
  rows: number,
  cols: number,
  initialValueOrFn: number | (() => number),
): NumericMatrix => {
  // return timesMapN(
  //   [rows, cols],
  //   typeof initialValueOrFn === 'function'
  //     ? initialValueOrFn
  //     : () => initialValueOrFn,
  // )

  const result: NumericMatrix = []

  for (let row = 0; row < rows; row++) {
    result[row] = []

    for (let col = 0; col < cols; col++) {
      result[row][col] =
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
 * @example createVector(3, 0) // => [0, 0, 0]
 */
export const createVector = (
  size: number,
  initialValueOrFn: number | (() => number),
): NumericVector => {
  // return timesMap(
  //   size,
  //   typeof initialValueOrFn === 'function'
  //     ? initialValueOrFn
  //     : () => initialValueOrFn,
  // )

  const result: NumericVector = []

  for (let i = 0; i < size; i++) {
    result[i] =
      typeof initialValueOrFn === 'function'
        ? initialValueOrFn()
        : initialValueOrFn
  }

  return result
}

/** Wraps a vector as a single-column matrix, so it can join matrix operations. */
export const toMatrix = (vector: NumericVector): NumericMatrix =>
  vector.map(value => [value])

/** Unpacks a single-column matrix back into a vector. */
export const fromMatrix = (matrix: NumericMatrix): NumericVector =>
  matrix.map(row => row[0])

/**
 * Element-wise matrix addition.
 *
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

  // return timesMapN(
  //   [rowsLeft, colsLeft],
  //   (row, col) => left[row][col] + right[row][col],
  // )

  const result: NumericMatrix = []

  for (let row = 0; row < rowsLeft; row++) {
    result[row] = []

    for (let col = 0; col < colsLeft; col++) {
      result[row][col] = left[row][col] + right[row][col]
    }
  }

  return result
}

/**
 * Standard matrix product: the rows of `left` dotted with the columns of
 * `right`. Used to compute z(𝓁) = w(𝓁)·a(𝓁-1).
 *
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

/** Swaps a matrix's rows and columns, reflecting it along its main diagonal. */
export const transposeMatrix = (matrix: NumericMatrix): NumericMatrix => {
  const cols = matrix[0].length
  const rows = matrix.length

  const result: NumericMatrix = []

  // timesForEachN([rows, cols], (row, col) => {
  //   result[col] ??= []
  //   result[col][row] = matrix[row][col]
  // })

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      result[col] ??= []
      result[col][row] = matrix[row][col]
    }
  }

  return result
}

/**
 * Multiplies every entry of a matrix by a scalar, used to scale a gradient by
 * the learning rate.
 */
export const multiplyMatrixByScalar = (
  matrix: NumericMatrix,
  scalar: number,
): NumericMatrix => {
  const rowsLeft = matrix.length
  const colsLeft = matrix[0]?.length ?? 0

  // return timesMapN(
  //   [rowsLeft, colsLeft],
  //   (row, col) => matrix[row][col] * scalar,
  // )

  const result: NumericMatrix = []

  for (let row = 0; row < rowsLeft; row++) {
    result[row] = []

    for (let col = 0; col < colsLeft; col++) {
      result[row][col] = matrix[row][col] * scalar
    }
  }

  return result
}

/**
 * Divides every entry of a matrix by a scalar, used to turn a summed batch
 * gradient into its mean.
 *
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

/** Multiplies every entry of a vector by a scalar. */
export const multiplyVectorByScalar = (
  vector: NumericVector,
  scalar: number,
): NumericVector => {
  return vector.map(value => value * scalar)
}

/**
 * Divides every entry of a vector by a scalar.
 *
 * @throws If `scalar` is 0.
 */
export const divideVectorByScalar = (
  vector: NumericVector,
  scalar: number,
): NumericVector => {
  if (scalar === 0) throw new Error('Cannot divide vector by 0')

  return multiplyVectorByScalar(vector, 1 / scalar)
}

/** A random number in [min, max). */
export const random = (min = 0, max = 1) => Math.random() * (max - min) + min

/**
 * Fisher-Yates shuffle, permuting the array in place, so each epoch sees the
 * data in a different order.
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
 * @example formatPercentageWithDecimalPlaces(0.985678, 2) // => 98.57
 */
export const formatPercentageWithDecimalPlaces = (
  value: number,
  decimalPlaces: number,
) => Math.round(value * 100 * 10 ** decimalPlaces) / 10 ** decimalPlaces

/**
 * Flattens every layer of a gradient into a single vector, so all of a
 * network's parameters can be handled as one list.
 *
 * Layers are ordered by 𝓁, each contributing its weights row-major then its
 * biases, so the result does not depend on the order they arrive in.
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
