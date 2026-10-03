import type { NumericMatrix, TrainingData } from './types.ts'
import { assertIsNotNull } from './utils.ts'
import { DATASET_PIXEL_MAX, imageToText } from './dataset.ts'

// MNIST dataset: constants describing the format, plus a loader for the
// gzipped IDX files bundled in ./data/mnist. Kept alongside the Animal-MNIST
// loader; both satisfy the DatasetLoader contract in dataset.ts.

/** IDX magic number expected at the start of every image file. */
export const MNIST_IMAGE_MAGIC = 0x00000803 // 2051

/** IDX magic number expected at the start of every label file. */
export const MNIST_LABEL_MAGIC = 0x00000801 // 2049

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
   * @returns Nothing; the parsed splits are stored on the loader.
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
   * Returns the name of a class index. The digit set has no name table, so the
   * index itself is the label.
   *
   * @param label Class index, in [0, 9].
   * @returns The digit as text, or the index itself when out of range.
   * @throws If the dataset has not been loaded.
   */
  className(label: number): string {
    assertIsNotNull(this.trainingData)
    return String(label)
  }

  /**
   * Reads a gzipped IDX image file. After decompression the header holds the
   * magic number, the image count, the rows and the columns (4 bytes each,
   * big-endian), followed by one byte per pixel, row-major.
   *
   * @param path Filesystem path of the gzipped IDX image file.
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
        image.push(decompressed[offset++] / DATASET_PIXEL_MAX)

      images.push(image)
    }

    return images
  }

  /**
   * Reads a gzipped IDX label file: the same 4-byte header of magic number and
   * count, then one byte per label.
   *
   * @param path Filesystem path of the gzipped IDX label file.
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

  /**
   * Decompresses gzip bytes with the built-in DecompressionStream API.
   *
   * @param data The compressed bytes.
   * @returns The decompressed bytes, concatenated in stream order.
   * @throws If the bytes are not valid gzip.
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
   * Renders one image from the loaded dataset as text, chosen by split and
   * index.
   *
   * @param type Which split to read from.
   * @param index Position of the image within that split.
   * @returns The image as text, one line per row of pixels.
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
   *
   * @param image One flattened image, row-major.
   * @returns The image as text, one line per row of pixels.
   */
  static imageToText(image: number[]): string {
    return imageToText(image)
  }
}
