import type { TrainingData } from './types.ts'
import { map, timesForEachN, assertIsNotNull } from './utils.ts'
import { NumericMatrix } from './types.ts'

// MNIST Dataset

/** Number of distinct digit classes in the dataset (0-9). */
export const MNIST_OUTPUT_SIZE = 10

/** Height in pixels of each MNIST image. */
export const MNIST_IMAGE_ROWS = 28

/** Width in pixels of each MNIST image. */
export const MNIST_IMAGE_COLS = 28

/**
 * IDX magic number expected at the start of every image file: 0x00000803
 * (2051), whose low byte 0x03 marks the 3-byte-label-plus-image variant.
 */
export const MNIST_IMAGE_MAGIC = 0x00000803 // 2051

/**
 * IDX magic number expected at the start of every label file: 0x00000801
 * (2049), whose low byte 0x01 marks the one-byte-label variant.
 */
export const MNIST_LABEL_MAGIC = 0x00000801 // 2049

/**
 * Largest value a raw pixel byte can hold, used to normalize pixels to
 * [0, 1].
 */
export const MNIST_PIXEL_MAX = 2 ** 8 - 1 // 255

/**
 * Characters used to render an image as text, ordered from darkest (index 0,
 * a space for a black pixel) to lightest (the final character for a white
 * pixel).
 */
const IMAGE_TO_TEXT_MAPPING = ' ░▒▓▉█'

/**
 * Locations of the four dataset files, gzipped and in Big Endian IDX format.
 * Read from disk rather than fetched over the network.
 */
const MNIST_PATHS = {
  trainImages: './data/mnist/train-images-idx3-ubyte.zip',
  trainLabels: './data/mnist/train-labels-idx1-ubyte.zip',
  testImages: './data/mnist/t10k-images-idx3-ubyte.zip',
  testLabels: './data/mnist/t10k-labels-idx1-ubyte.zip',
}

/**
 * Reads and parses the MNIST dataset, exposed as {@link TrainingData} in
 * `trainingData` / `testData`, from the local .zip files bundled in
 * `./data/mnist`.
 *
 * The raw files are gzipped IDX binaries; pixels are normalized to [0, 1].
 */
export class MnistLoader {
  /**
   * The parsed training split, or `null` until {@link MnistLoader.load}
   * succeeds. `inputs` holds one flattened 784-value image per row and
   * `labels` the matching digit for each.
   */
  trainingData: TrainingData | null = null

  /**
   * The parsed test split, shaped like {@link MnistLoader.trainingData}, or `null`
   * until {@link MnistLoader.load} succeeds.
   */
  testData: TrainingData | null = null

  /**
   * Whether the dataset has already been read, used to make
   * {@link MnistLoader.load} idempotent. Reading ~60 MB of IDX data twice is
   * wasted work, so repeat calls return immediately.
   */
  loaded = false

  /**
   * Reads and parses all four gzipped IDX files and stores the results in
   * {@link MnistLoader.trainingData} and {@link MnistLoader.testData}.
   *
   * Does nothing when {@link MnistLoader.loaded} is already true, so it is safe
   * to call more than once.
   *
   * @param onProgress Optional callback invoked with human-readable status
   * messages as each file is read, for progress display. Omit it to stay quiet.
   * @returns Nothing; inspect `trainingData` / `testData` afterwards.
   * @throws If any file is missing or unreadable, if a file is not valid gzip,
   * or if a magic number does not match the expected images/labels constant.
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
   * Reads and parses MNIST image data from a gzipped IDX binary file on disk.
   *
   * File format (after decompression):
   *   Bytes 0-3:   Magic number (MNIST_IMAGE_MAGIC = 0x00000803 = 2051 for images)
   *   Bytes 4-7:   Number of images
   *   Bytes 8-11:  Number of rows per image (MNIST_IMAGE_DIMENSIONS.rows = 28)
   *   Bytes 12-15: Number of columns per image (MNIST_IMAGE_DIMENSIONS.cols = 28)
   *   Bytes 16+:   Pixel values (0-255), one byte per pixel, row-major order
   *
   * Each pixel is normalized to [0, 1] by dividing by 255.
   * Returns a 2D array where each inner array is a flattened 28×28 image (784 values).
   *
   * @param path Path to the gzipped IDX file to read.
   * @returns A 2D array with one inner array per image, each holding 784 values
   * normalized to [0, 1] in row-major order.
   * @throws If the file cannot be read, is not valid gzip, or its magic number
   * is not {@link MNIST_IMAGE_MAGIC}.
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

    const images: number[][] = []
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
   * Reads and parses MNIST label data from a gzipped IDX binary file on disk.
   *
   * File format (after decompression):
   *   Bytes 0-3: Magic number (MNIST_LABEL_MAGIC = 0x00000801 = 2049 for labels)
   *   Bytes 4-7: Number of labels
   *   Bytes 8+:  Label values (0-9), one byte per label
   *
   * @param path Path to the gzipped IDX file to read.
   * @returns A flat array with one digit (0-9) per label, in file order.
   * @throws If the file cannot be read, is not valid gzip, or its magic number
   * is not {@link MNIST_LABEL_MAGIC}.
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

  /**
   * Decompresses gzip data using the built-in DecompressionStream API.
   *
   * The stream is drained concurrently with writing, so output chunks are
   * buffered and concatenated once the total length is known.
   *
   * @param data The compressed bytes.
   * @returns A single `Uint8Array` holding the decompressed bytes.
   * @throws If the input is not valid gzip data.
   */
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
   * Convenience wrapper over {@link MnistLoader.imageToText} that retrieves
   * the image from the loaded dataset by type and index.
   *
   * @param type - 'trainingData' for training images, 'testData' for test images
   * @param index - Index of the image within the selected dataset (0-based)
   * @returns A string containing the 28×28 image rendered as text with newline-separated rows
   * @throws If data has not been loaded (call `load()` first) or index is out of bounds
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
   * Renders a 28×28 image (flattened to 784 values in [0, 1]) as text
   * using Unicode block characters for visual density.
   *
   * Each pixel is mapped to a character from the mapping ` ░▒▓▉█`,
   * where 0 (black) maps to a space and 1 (white) maps to `█`.
   * Characters are doubled horizontally to approximate square pixels
   * in monospace fonts.
   *
   * @param image - Flattened 28×28 image as an array of 784 values in [0, 1]
   * @returns A string with 28 newline-separated rows, each 56 characters wide
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
