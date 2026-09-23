import type { TrainingData } from './types.ts'
import { map, timesForEachN, timesMap } from './utils.ts'
import {
  MNIST_IMAGE_ROWS,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_MAGIC,
  MNIST_LABEL_MAGIC,
  MNIST_PIXEL_MAX,
} from './constants.ts'

// Files in Big Endian IDX format, gzipped.
const MNIST_PATHS = {
  trainImages: './mnist/train-images-idx3-ubyte.zip',
  trainLabels: './mnist/train-labels-idx1-ubyte.zip',
  testImages: './mnist/t10k-images-idx3-ubyte.zip',
  testLabels: './mnist/t10k-labels-idx1-ubyte.zip',
}

export class MnistLoader {
  private trainData: TrainingData | null = null
  private testData: TrainingData | null = null
  private loaded = false

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

    this.trainData = { inputs: trainImages, labels: trainLabels }
    this.testData = { inputs: testImages, labels: testLabels }
    this.loaded = true

    onProgress?.(
      `Loaded ${trainImages.length} training and ${testImages.length} test samples`,
    )
  }

  getTrainingData(): TrainingData {
    if (!this.trainData) throw new Error('Data not loaded. Call load() first.')

    return this.trainData
  }

  getTestData(): TrainingData {
    if (!this.testData) throw new Error('Data not loaded. Call load() first.')

    return this.testData
  }

  isLoaded(): boolean {
    return this.loaded
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
   */
  private async downloadAndParseImages(path: string): Promise<number[][]> {
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
   * Preprocesses canvas ImageData to match MNIST format.
   *
   * 1. Downsamples the canvas to 28×28 by nearest-neighbor pixel sampling
   * 2. Computes the center of mass of the drawn digit (intensity-weighted)
   * 3. Translates the digit so its center of mass aligns with the grid center
   * 4. Normalizes pixel values from [0, 255] to [0, 1]
   *
   * Centering ensures the network receives consistently positioned digits
   * regardless of where they are drawn on the canvas.
   *
   * @returns A flat array of 784 values (28×28), ready for network input.
   * The canvas draws white digits on black background, matching MNIST's format.
   */
  preprocessCanvasData(imageData: ImageData): number[] {
    const { width, height, data } = imageData
    const stepX = width / MNIST_IMAGE_COLS
    const stepY = height / MNIST_IMAGE_ROWS

    // Step 1: Downsample to 28×28
    const grid: number[][] = timesMap(MNIST_IMAGE_ROWS, () =>
      new Array(MNIST_IMAGE_COLS).fill(0),
    )

    // Average every pixel within each 10×10 cell so thin strokes
    // that fall between sample points are never missed.
    timesForEachN([MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS], (y, x) => {
      const x0 = Math.floor(x * stepX)
      const y0 = Math.floor(y * stepY)
      const x1 = Math.floor((x + 1) * stepX)
      const y1 = Math.floor((y + 1) * stepY)

      let sum = 0
      let count = 0

      for (let sy = y0; sy < y1; sy++) {
        for (let sx = x0; sx < x1; sx++) {
          const idx = (sy * width + sx) * 4

          sum += data[idx] // red channel (grayscale, so r ≈ g ≈ b)
          count++
        }
      }

      grid[y][x] = Math.trunc(sum / count)
    })

    // Step 2: Find bounding box of the digit
    let minX = MNIST_IMAGE_COLS,
      maxX = -1,
      minY = MNIST_IMAGE_ROWS,
      maxY = -1

    timesForEachN([MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS], (y, x) => {
      if (grid[y][x] > 0) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    })

    // If no digit is drawn, return blank.
    if (minX > maxX)
      return new Array(MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS).fill(0)

    // Step 3: Compute the intensity-weighted center of mass of the digit.
    //
    // A naive sumX += x * w would zero out the contribution of any pixel
    // on the first column (x=0) or first row (y=0), effectively ignoring
    // them. Shifting to 1-based coordinates before the multiply avoids
    // this, and subtracting 1 afterwards restores the original scale.
    //
    // The offset (dx, dy) is the integer translation needed to move the
    // center of mass to the grid center (14, 14).
    let sumX = 0,
      sumY = 0,
      totalWeight = 0

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const w = grid[y][x]

        if (w > 0) {
          sumX += (x + 1) * w
          sumY += (y + 1) * w

          totalWeight += w
        }
      }
    }

    const centerOfMassX = Math.ceil(sumX / totalWeight - 1)
    const centerOfMassY = Math.ceil(sumY / totalWeight - 1)
    const centerX = MNIST_IMAGE_COLS / 2
    const centerY = MNIST_IMAGE_ROWS / 2
    const dx = Math.round(centerX - centerOfMassX)
    const dy = Math.round(centerY - centerOfMassY)

    // Step 4: Translate to center the digit
    const centered: number[][] = timesMap(MNIST_IMAGE_ROWS, () =>
      new Array(MNIST_IMAGE_COLS).fill(0),
    )

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const newY = y + dy
        const newX = x + dx

        if (
          newY >= 0 &&
          newY < MNIST_IMAGE_ROWS &&
          newX >= 0 &&
          newX < MNIST_IMAGE_COLS
        )
          centered[newY][newX] = grid[y][x]
      }
    }

    // Step 5: Flatten and normalize
    const image: number[] = []

    timesForEachN([MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS], (y, x) => {
      image.push(centered[y][x] / MNIST_PIXEL_MAX)
    })

    console.log(MnistLoader.imageToText(image))

    return image
  }

  /**
   * Convenience wrapper over {@link MnistLoader.imageToText} that retrieves
   * the image from the loaded dataset by type and index.
   *
   * @param type - 'trainData' for training images, 'testData' for test images
   * @param index - Index of the image within the selected dataset (0-based)
   * @returns A string containing the 28×28 image rendered as text with newline-separated rows
   * @throws If data has not been loaded (call `load()` first) or index is out of bounds
   */
  imageAsText(type: 'trainData' | 'testData', index: number): string {
    return MnistLoader.imageToText(
      type === 'trainData'
        ? this.getTrainingData().inputs[index]
        : this.getTestData().inputs[index],
    )
  }

  /**
   * Renders a 28×28 image (flattened to 784 values in [0, 1]) as text
   * using Unicode block characters for visual density.
   *
   * Each pixel is mapped to a character from the gradient ` ░▒▓▉█`,
   * where 0 (black) maps to a space and 1 (white) maps to `█`.
   * Characters are doubled horizontally to approximate square pixels
   * in monospace fonts.
   *
   * @param image - Flattened 28×28 image as an array of 784 values in [0, 1]
   * @returns A string with 28 newline-separated rows, each 56 characters wide
   */
  static imageToText(image: number[]): string {
    const gradient = ' ░▒▓▉█'

    let text = ''

    timesForEachN([MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS], (row, col) => {
      const pixelIndex = row * MNIST_IMAGE_COLS + col
      const value = image[pixelIndex]
      const charIndex = Math.trunc(
        map(value, 0, 1, 0, gradient.length - 1, true),
      )
      const unicodeChar = gradient[charIndex]

      text += unicodeChar.repeat(2)

      if (col === MNIST_IMAGE_COLS - 1) text += '\n'
    })

    return text
  }
}
