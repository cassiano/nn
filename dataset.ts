import type { TrainingData } from './types.ts'
import { map, timesForEachN } from './utils.ts'

// Shared dataset contract. Both loaders — the handwritten-digit IDX reader and
// the Animal-MNIST .npz reader — describe the same 28x28 grayscale, 10-class
// problem, so the geometry and the ASCII renderer live here rather than being
// duplicated per format. Format-specific constants stay next to their loader.

// Which dataset to train on. 'digits' is the historical handwritten-digit set.
export type DatasetChoice = 'digits' | 'animal'

/** Number of distinct classes in either dataset (0-9). */
export const DATASET_OUTPUT_SIZE = 10

/** Height in pixels of each image. */
export const DATASET_IMAGE_ROWS = 28

/** Width in pixels of each image. */
export const DATASET_IMAGE_COLS = 28

/** Largest value a raw pixel byte can hold, used to normalize pixels to [0, 1]. */
export const DATASET_PIXEL_MAX = 2 ** 8 - 1 // 255

/** Total number of images in Animal-MNIST (the digit set ships 70k). */
export const DATASET_TOTAL_SAMPLES = {
  digits: 70_000,
  animal: 10_000,
} as const

/**
 * Characters used to render an image as text, ordered from darkest to lightest.
 */
export const IMAGE_TO_TEXT_MAPPING = ' ░▒▓▉█'

/**
 * The behaviour `main.ts` relies on, satisfied by every loader.
 */
export type DatasetLoader = {
  /** The parsed training split, or `null` until `load` runs. */
  trainingData: TrainingData | null
  /** The parsed test split, shaped like `trainingData`. */
  testData: TrainingData | null
  /** Whether the dataset has been read, so loading twice is skipped. */
  loaded: boolean
  /** Reads the dataset, reporting progress through the optional callback. */
  load: (onProgress?: (msg: string) => void) => Promise<void>
  /** Renders one image from the chosen split as text. */
  imageAsText: (type: 'trainingData' | 'testData', index: number) => string
  /** The human-readable name of a class index. */
  className: (label: number) => string
}

/** Usage text printed for -h/--help and on an unrecognized flag. */
export const DATASET_FLAG_USAGE = `Usage: deno run --allow-read main.ts [-a | -d]

  -d, --digits  handwritten MNIST digits from ./data/mnist (default)
  -a, --animal  Animal-MNIST silhouettes from ./data/animal-mnist
  -h, --help    show this message`

/**
 * Reports whether the run should only print usage.
 *
 * @param args The command line arguments, usually `Deno.args`.
 * @returns True when -h or --help is present.
 */
export function wantsHelp(args: readonly string[]): boolean {
  return args.includes('-h') || args.includes('--help')
}

/**
 * Resolves which dataset to load from the command line.
 *
 * @param args The command line arguments, usually `Deno.args`.
 * @returns The requested dataset, defaulting to 'digits'.
 * @throws If an argument is not a recognized dataset flag.
 */
export function parseDatasetFlag(args: readonly string[]): DatasetChoice {
  let choice: DatasetChoice = 'digits'

  // Every argument is validated, so trailing junk is rejected rather than
  // silently dropped after the first recognized flag.
  for (const arg of args) {
    if (arg === '-a' || arg === '--animal') choice = 'animal'
    else if (arg === '-d' || arg === '--digits' || arg === '--mnist')
      choice = 'digits'
    else throw new Error(`Unknown argument "${arg}".\n${DATASET_FLAG_USAGE}`)
  }

  return choice
}

/**
 * Renders a flattened image as text, mapping each pixel to a character from
 * ` ░▒▓▉█` and doubling it horizontally to approximate square pixels.
 *
 * @param image One flattened image, row-major, normalized to [0, 1].
 * @returns The image as text, one line per row of pixels.
 */
export function imageToText(image: number[]): string {
  let text = ''

  timesForEachN([DATASET_IMAGE_ROWS, DATASET_IMAGE_COLS], (row, col) => {
    const pixelIndex = row * DATASET_IMAGE_COLS + col
    const value = image[pixelIndex]
    const charIndex = Math.trunc(
      map(value, 0, 1, 0, IMAGE_TO_TEXT_MAPPING.length - 1, true),
    )
    const unicodeChar = IMAGE_TO_TEXT_MAPPING[charIndex]

    text += unicodeChar.repeat(2)

    if (col === DATASET_IMAGE_COLS - 1) text += '\n'
  })

  return text
}