import type { TrainingData } from './types.ts'
import { map, timesForEachN, assertIsNotNull } from './utils.ts'
import { NumericMatrix } from './types.ts'

// MNIST dataset: constants describing the format, plus a loader for the
// gzipped IDX files bundled in ./data/mnist.

/** Number of distinct digit classes in the dataset (0-9). */
export const MNIST_OUTPUT_SIZE = 10

/** Height in pixels of each MNIST image. */
export const MNIST_IMAGE_ROWS = 28

/** Width in pixels of each MNIST image. */
export const MNIST_IMAGE_COLS = 28

/** IDX magic number expected at the start of every image file. */
export const MNIST_IMAGE_MAGIC = 0x00000803 // 2051

/** IDX magic number expected at the start of every label file. */
export const MNIST_LABEL_MAGIC = 0x00000801 // 2049

/** Largest value a raw pixel byte can hold, used to normalize pixels to [0, 1]. */
export const MNIST_PIXEL_MAX = 2 ** 8 - 1 // 255

/**
 * Characters used to render an image as text, ordered from darkest to lightest.
 */
const IMAGE_TO_TEXT_MAPPING = ' ░▒▓▉█'

/** The four dataset files, gzipped and in Big Endian IDX format. */
const MNIST_PATHS = {
  trainImages: './data/mnist/train-images-idx3-ubyte.zip',
  trainLabels: './data/mnist/train-labels-idx1-ubyte.zip',
  testImages: './data/mnist/t10k-images-idx3-ubyte.zip',
  testLabels: './data/mnist/t10k-labels-idx1-ubyte.zip',
}

/**
 * Reads the MNIST dataset from the local gzipped IDX files into
 * `trainingData` / `testData`, with pixels normalized to [0, 1].
 */
export class MnistLoader {
  /** The parsed training split, or `null` until {@link MnistLoader.load} runs. */
  trainingData: TrainingData | null = null

  /** The parsed test split, shaped like {@link MnistLoader.trainingData}. */
  testData: TrainingData | null = null

  /** Whether the dataset has been read, so loading twice is skipped. */
  loaded = false

  /**
   * Reads and parses all four files into {@link MnistLoader.trainingData} and
   * {@link MnistLoader.testData}. Safe to call more than once.
   *
   * @param onProgress Called with status messages as each file is read.
   * @throws If a file is missing, is not valid gzip, or has the wrong magic number.
   */
  async load(onProgress?: (msg: string) => void): Promise<void> {
    if (this.loaded) return

    onProgress?.('Loading MNIST training images...')
    const trainImages = await this.downloadAndParseImages(
      MNIST_PATHS.trainImages,
    )

    onProgress?.('Loading MNIST training labels...')
    const trainLabels = await this.downloadAndParseLabels(
      MNIST_PATHS.trainLabels,
    )

    onProgress?.('Loading MNIST test images...')
    const testImages = await this.downloadAndParseImages(MNIST_PATHS.testImages)

    onProgress?.('Loading MNIST test labels...')
    const testLabels = await this.downloadAndParseLabels(MNIST_PATHS.testLabels)

    this.trainingData = { inputs: trainImages, labels: trainLabels }
    this.testData = { inputs: testImages, labels: testLabels }
    this.loaded = true

    onProgress?.(
      `Loaded ${trainImages.length} training and ${testImages.length} test samples`,
    )
  }

  /**
   * Reads a gzipped IDX image file. After decompression the header holds the
   * magic number, the image count, the rows and the columns (4 bytes each,
   * big-endian), followed by one byte per pixel, row-major.
   *
   * @returns One flattened image per row, values normalized to [0, 1].
   * @throws If the file cannot be read, is not valid gzip, or is not an image file.
   */
  private async downloadAndParseImages(path: string): Promise<NumericMatrix> {
    const data = await Deno.readFile(path)
    const decompressed = await this.gunzip(data)
    const view = new DataView(decompressed.buffer) // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView

    const magic = view.getUint32(0, false)
    if (magic !== MNIST_IMAGE_MAGIC)
      throw new Error(`Invalid magic number for images: ${magic}`)

    const count = view.getUint32(4, false)
    const rows = view.getUint32(8, false)
    const cols = view.getUint32(12, false)

    const images: NumericMatrix = []
    let offset = 16

    for (let i = 0; i < count; i++) {
      const image: number[] = []

      for (let j = 0; j < rows * cols; j++)
        image.push(decompressed[offset++] / MNIST_PIXEL_MAX)

      images.push(image)
    }

    return images
  }

  /**
   * Reads a gzipped IDX label file: the same 4-byte header of magic number and
   * count, then one byte per label.
   *
   * @returns One digit per label, in file order.
   * @throws If the file cannot be read, is not valid gzip, or is not a label file.
   */
  private async downloadAndParseLabels(path: string): Promise<number[]> {
    const data = await Deno.readFile(path)
    const decompressed = await this.gunzip(data)
    const view = new DataView(decompressed.buffer) // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView

    const magic = view.getUint32(0, false)
    if (magic !== MNIST_LABEL_MAGIC)
      throw new Error(`Invalid magic number for labels: ${magic}`)

    const count = view.getUint32(4, false)
    const labels: number[] = []

    for (let i = 0; i < count; i++) labels.push(decompressed[8 + i])

    return labels
  }

  /** Decompresses gzip bytes with the built-in DecompressionStream API. */
  private async gunzip(data: Uint8Array): Promise<Uint8Array> {
    const ds = new DecompressionStream('gzip')
    const writer = ds.writable.getWriter()
    const reader = ds.readable.getReader()

    const chunks: Uint8Array[] = []
    let totalLength = 0

    const readTask = (async () => {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        chunks.push(value)
        totalLength += value.length
      }
    })()

    await writer.write(data.buffer as ArrayBuffer)
    await writer.close()
    await readTask

    const result = new Uint8Array(totalLength)
    let offset = 0

    for (const chunk of chunks) {
      result.set(chunk, offset)

      offset += chunk.length
    }

    return result
  }

  /**
   * Renders one image from the loaded dataset as text, chosen by split and
   * index.
   *
   * @throws If the dataset has not been loaded, or the index is out of bounds.
   */
  imageAsText(type: 'trainingData' | 'testData', index: number): string {
    assertIsNotNull(this.trainingData)
    assertIsNotNull(this.testData)

    return MnistLoader.imageToText(
      type === 'trainingData'
        ? this.trainingData.inputs[index]
        : this.testData.inputs[index],
    )
  }

  /**
   * Renders a flattened image as text, mapping each pixel to a character from
   * ` ░▒▓▉█` and doubling it horizontally to approximate square pixels.
   */
  static imageToText(image: number[]): string {
    let text = ''

    timesForEachN([MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS], (row, col) => {
      const pixelIndex = row * MNIST_IMAGE_COLS + col
      const value = image[pixelIndex]
      const charIndex = Math.trunc(
        map(value, 0, 1, 0, IMAGE_TO_TEXT_MAPPING.length - 1, true),
      )
      const unicodeChar = IMAGE_TO_TEXT_MAPPING[charIndex]

      text += unicodeChar.repeat(2)

      if (col === MNIST_IMAGE_COLS - 1) text += '\n'
    })

    return text
  }
}
