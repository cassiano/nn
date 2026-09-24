import { NumericMatrix } from './types.ts'
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

  return (
    ((value - lower) / (higher - lower)) * (projectedUpper - projectedLower) +
    projectedLower
  )
}

export const timesForEach = (count: number, fn: (i: number) => void) => {
  for (let i = 0; i < count; i++) fn(i)
}

export const timesMap = <T>(count: number, fn: (index: number) => T): T[] => {
  const results: T[] = []

  for (let i = 0; i < count; i++) results[i] = fn(i)

  return results
}

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

export const toMatrix = (vector: number[]): number[][] =>
  vector.map(value => [value])
export const fromMatrix = (matrix: number[][]): number[] =>
  matrix.map(row => row[0])

export const addMatrices = (
  left: NumericMatrix,
  right: NumericMatrix,
): NumericMatrix => {
  const colsLeft = left[0].length
  const colsRight = right[0].length
  const rowsLeft = left.length
  const rowsRight = right.length

  if (colsLeft !== colsRight)
    throw new Error(
      `Number of columns from left matrix (${colsLeft}) must match number of columns from right one (${colsRight})`,
    )

  if (rowsLeft !== rowsRight)
    throw new Error(
      `Number of rows from left matrix (${rowsLeft}) must match number of rows from right one (${rowsRight})`,
    )

  const result: NumericMatrix = []

  for (let row = 0; row < rowsLeft; row++) {
    result[row] = []

    for (let col = 0; col < colsRight; col++)
      result[row][col] = left[row][col] + right[row][col]
  }

  return result
}

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
