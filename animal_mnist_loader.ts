import type { NumericMatrix, TrainingData } from './types.ts'
import { assertIsNotNull } from './utils.ts'
import {
  DATASET_IMAGE_COLS,
  DATASET_IMAGE_ROWS,
  DATASET_PIXEL_MAX,
  imageToText,
} from './dataset.ts'

// Animal-MNIST dataset: constants describing the format, plus a loader for the
// .npz bundle in ./data/animal-mnist. The bundle is a ZIP archive holding NumPy
// .npy members (X, y, class_names) rather than the gzipped IDX files MNIST uses.

/** Total number of images in the dataset. */
export const ANIMAL_SAMPLE_COUNT = 10_000

/** The class each label index depicts, in index order. */
export const ANIMAL_CLASS_NAMES = [
  'Bear',
  'Bird',
  'Cat',
  'Cow',
  'Dog',
  'Elephant',
  'Giraffe',
  'Horse',
  'Sheep',
  'Zebra',
] as const

/** Location of the .npz bundle, relative to the working directory. */
export const ANIMAL_NPZ_PATH = './data/animal-mnist/animal_mnist.npz'

/** Fraction of the dataset held out as the test split when none is given. */
export const ANIMAL_DEFAULT_TEST_FRACTION = 0.2

/** Seed for the shuffle that interleaves classes before splitting. */
export const ANIMAL_SHUFFLE_SEED = 0x5eed

/** The `\x93NUMPY` magic number every .npy member starts with. */
const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]

/** ZIP signatures, read from the central directory rather than local headers. */
const ZIP_EOCD_SIGNATURE = 0x06054b50
const ZIP_CENTRAL_SIGNATURE = 0x02014b50
const ZIP_LOCAL_SIGNATURE = 0x04034b50

/** One parsed .npy member: its dtype string, shape and a view over its data. */
type NpyArray = {
  descr: string
  fortranOrder: boolean
  shape: number[]
  data: DataView
}

/** How {@link AnimalMnistLoader} reads and splits the dataset. */
export type AnimalLoaderOptions = {
  /** Filesystem path of the .npz bundle. */
  path?: string
  /** Share of the dataset reserved for the test split, in [0, 1). */
  testFraction?: number
  /** Seed for the deterministic shuffle applied before splitting. */
  seed?: number
}

/**
 * Reads the Animal-MNIST dataset from the local .npz bundle into
 * `trainingData` / `testData`, with pixels normalized to [0, 1].
 */
export class AnimalMnistLoader {
  /** The parsed training split, or `null` until {@link AnimalMnistLoader.load} runs. */
  trainingData: TrainingData | null = null

  /** The parsed test split, shaped like {@link AnimalMnistLoader.trainingData}. */
  testData: TrainingData | null = null

  /** Whether the dataset has been read, so loading twice is skipped. */
  loaded = false

  /** The class names read from the bundle, or `null` before loading. */
  classNames: string[] | null = null

  /** Filesystem path of the .npz bundle. */
  private readonly path: string

  /** Share of the dataset reserved for the test split. */
  private readonly testFraction: number

  /** Seed for the shuffle that interleaves classes before splitting. */
  private readonly seed: number

  /**
   * @param options Overrides for the bundle path, split ratio and shuffle seed.
   */
  constructor(options: AnimalLoaderOptions = {}) {
    this.path = options.path ?? ANIMAL_NPZ_PATH
    this.testFraction = options.testFraction ?? ANIMAL_DEFAULT_TEST_FRACTION
    this.seed = options.seed ?? ANIMAL_SHUFFLE_SEED
  }

  /**
   * Reads and parses the bundle into {@link AnimalMnistLoader.trainingData} and
   * {@link AnimalMnistLoader.testData}. Safe to call more than once.
   *
   * @param onProgress Called with status messages as the bundle is read.
   * @returns Nothing; the parsed splits are stored on the loader.
   * @throws If the bundle is missing, is not a ZIP, or lacks the expected members.
   */
  async load(onProgress?: (msg: string) => void): Promise<void> {
    if (this.loaded) return

    onProgress?.('Loading Animal-MNIST images...')
    const archive = await Deno.readFile(this.path)
    const entries = await AnimalMnistLoader.readZipEntries(archive)

    const imagesArray = AnimalMnistLoader.parseNpy(
      AnimalMnistLoader.requireEntry(entries, 'X.npy'),
    )
    const labelsArray = AnimalMnistLoader.parseNpy(
      AnimalMnistLoader.requireEntry(entries, 'y.npy'),
    )
    const namesArray = AnimalMnistLoader.parseNpy(
      AnimalMnistLoader.requireEntry(entries, 'class_names.npy'),
    )

    const [count, rows, cols] = imagesArray.shape
    if (labelsArray.shape[0] !== count)
      throw new Error(
        `Image count ${count} does not match label count ${labelsArray.shape[0]}`,
      )
    if (rows !== DATASET_IMAGE_ROWS || cols !== DATASET_IMAGE_COLS)
      throw new Error(
        `Expected ${DATASET_IMAGE_ROWS}x${DATASET_IMAGE_COLS} images, got ${rows}x${cols}`,
      )

    onProgress?.('Parsing images and labels...')
    const allImages = AnimalMnistLoader.parseImages(
      imagesArray,
      count,
      rows * cols,
    )
    const allLabels = AnimalMnistLoader.parseLabels(labelsArray)
    this.classNames = AnimalMnistLoader.parseStrings(namesArray)

    onProgress?.('Splitting into training and test sets...')
    const order = AnimalMnistLoader.shuffledIndices(count, this.seed)
    const testCount = Math.round(count * this.testFraction)
    const testOrder = order.slice(0, testCount)
    const trainOrder = order.slice(testCount)

    this.trainingData = {
      inputs: trainOrder.map(i => allImages[i]),
      labels: trainOrder.map(i => allLabels[i]),
    }
    this.testData = {
      inputs: testOrder.map(i => allImages[i]),
      labels: testOrder.map(i => allLabels[i]),
    }
    this.loaded = true

    onProgress?.(
      `Loaded ${this.trainingData.inputs.length} training and ${this.testData.inputs.length} test samples`,
    )
  }

  /**
   * Returns the class name a label index depicts.
   *
   * @param label Label index, in [0, 9].
   * @returns The class name, or the index itself when out of range.
   * @throws If the dataset has not been loaded.
   */
  className(label: number): string {
    assertIsNotNull(this.classNames)
    return this.classNames[label] ?? String(label)
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

    return AnimalMnistLoader.imageToText(
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

  /**
   * Reads every member of a ZIP archive, keyed by file name.
   *
   * Walks the central directory, then follows each local header to its payload,
   * inflating deflated members and copying stored ones verbatim.
   *
   * @param archive The raw archive bytes.
   * @returns The decompressed bytes of each member.
   * @throws If the central directory is missing or a member cannot be read.
   */
  private static async readZipEntries(
    archive: Uint8Array<ArrayBuffer>,
  ): Promise<Map<string, Uint8Array<ArrayBuffer>>> {
    const view = new DataView(
      archive.buffer,
      archive.byteOffset,
      archive.byteLength,
    ) // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView
    const entries = new Map<string, Uint8Array<ArrayBuffer>>()

    const eocd = AnimalMnistLoader.findEndOfCentralDirectory(view)
    const entryCount = view.getUint16(eocd + 10, true)
    let cursor = view.getUint32(eocd + 16, true)

    for (let i = 0; i < entryCount; i++) {
      if (view.getUint32(cursor, true) !== ZIP_CENTRAL_SIGNATURE)
        throw new Error(`Corrupt ZIP central directory at byte ${cursor}`)

      const method = view.getUint16(cursor + 10, true)
      const compressedSize = view.getUint32(cursor + 20, true)
      const nameLength = view.getUint16(cursor + 28, true)
      const extraLength = view.getUint16(cursor + 30, true)
      const commentLength = view.getUint16(cursor + 32, true)
      const localOffset = view.getUint32(cursor + 42, true)
      const name = new TextDecoder().decode(
        archive.subarray(cursor + 46, cursor + 46 + nameLength),
      )

      if (view.getUint32(localOffset, true) !== ZIP_LOCAL_SIGNATURE)
        throw new Error(`Corrupt ZIP local header for ${name}`)

      // The local header repeats the name, and may carry extra fields of its own.
      const localNameLength = view.getUint16(localOffset + 26, true)
      const localExtraLength = view.getUint16(localOffset + 28, true)
      const start = localOffset + 30 + localNameLength + localExtraLength
      const payload = archive.subarray(start, start + compressedSize)

      entries.set(
        name,
        method === 0 ? payload : await AnimalMnistLoader.inflateRaw(payload),
      )

      cursor += 46 + nameLength + extraLength + commentLength
    }

    return entries
  }

  /**
   * Scans backwards for the end-of-central-directory record, which sits last
   * but may be preceded by an arbitrary-length archive comment.
   *
   * @param view A DataView over the whole archive.
   * @returns The byte offset of the record.
   * @throws If no end-of-central-directory record is found.
   */
  private static findEndOfCentralDirectory(view: DataView): number {
    // 22 fixed bytes plus the largest possible 65535-byte comment.
    const earliest = Math.max(0, view.byteLength - (22 + 0xffff))

    for (let i = view.byteLength - 22; i >= earliest; i--)
      if (view.getUint32(i, true) === ZIP_EOCD_SIGNATURE) return i

    throw new Error('Not a ZIP archive: no end-of-central-directory record')
  }

  /**
   * Decompresses raw DEFLATE bytes with the built-in DecompressionStream API.
   *
   * @param data The compressed bytes.
   * @returns The decompressed bytes, concatenated in stream order.
   * @throws If the bytes are not valid DEFLATE data.
   */
  private static async inflateRaw(data: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
    const ds = new DecompressionStream('deflate-raw')
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

    await writer.write(data)
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
   * Fetches a required member, so a renamed file fails loudly.
   *
   * @param entries The archive members, keyed by name.
   * @param name The member to fetch.
   * @returns The member's bytes.
   * @throws If the member is absent.
   */
  private static requireEntry(
    entries: Map<string, Uint8Array<ArrayBuffer>>,
    name: string,
  ): Uint8Array<ArrayBuffer> {
    const entry = entries.get(name)
    if (entry === undefined) throw new Error(`Missing ${name} in the npz bundle`)

    return entry
  }

  /**
   * Parses the header of a .npy member, locating where its array data begins.
   *
   * @param bytes The member's bytes.
   * @returns The dtype string, shape and a view over the array data.
   * @throws If the magic number or header is malformed.
   */
  private static parseNpy(bytes: Uint8Array): NpyArray {
    for (let i = 0; i < NPY_MAGIC.length; i++)
      if (bytes[i] !== NPY_MAGIC[i])
        throw new Error('Not a .npy member: bad magic number')

    const view = new DataView(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength,
    ) // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView
    const majorVersion = view.getUint8(6)
    // v1 stores the header length in 2 bytes, v2 and v3 in 4.
    const headerLength = majorVersion === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
    const headerStart = (majorVersion === 1 ? 10 : 12) + headerLength
    const header = new TextDecoder().decode(bytes.subarray(10, headerStart - 2))

    const descr = header.match(/'descr'\s*:\s*'([^']+)'/)?.[1]
    if (descr === undefined) throw new Error('.npy header has no dtype')

    const shape = (header.match(/'shape'\s*:\s*\(([^)]*)\)/)?.[1] ?? '')
      .split(',')
      .map(part => part.trim())
      .filter(part => part.length > 0)
      .map(Number)

    const fortranOrder =
      header.match(/'fortran_order'\s*:\s*(True|False)/)?.[1] === 'True'

    return {
      descr,
      fortranOrder,
      shape,
      data: new DataView(bytes.buffer, bytes.byteOffset + headerStart),
    }
  }

  /**
   * Reads uint8 images, normalizing each pixel to [0, 1].
   *
   * @param array The parsed X member.
   * @param count How many images to read.
   * @param pixelsPerImage Row-major pixel count of a single image.
   * @returns One flattened image per row, values normalized to [0, 1].
   * @throws If the member is not uint8 or is C-order-agnostic Fortran data.
   */
  private static parseImages(
    array: NpyArray,
    count: number,
    pixelsPerImage: number,
  ): NumericMatrix {
    if (array.descr !== '|u1' && array.descr !== '<u1' && array.descr !== 'u1')
      throw new Error(`Expected uint8 images, got dtype ${array.descr}`)
    if (array.fortranOrder) throw new Error('Fortran-ordered images are not supported')

    const images: NumericMatrix = []

    for (let i = 0; i < count; i++) {
      const image: number[] = new Array(pixelsPerImage)
      const start = i * pixelsPerImage

      for (let j = 0; j < pixelsPerImage; j++)
        image[j] = array.data.getUint8(start + j) / DATASET_PIXEL_MAX

      images.push(image)
    }

    return images
  }

  /**
   * Reads int64 labels. Values are read via BigInt so the 8-byte width is
   * honoured, then narrowed, since class indices are small.
   *
   * @param array The parsed y member.
   * @returns One label index per image, in file order.
   * @throws If the member is not little-endian int64.
   */
  private static parseLabels(array: NpyArray): number[] {
    if (array.descr !== '<i8') throw new Error(`Expected int64 labels, got dtype ${array.descr}`)

    const labels: number[] = new Array(array.shape[0])

    for (let i = 0; i < labels.length; i++)
      labels[i] = Number(array.data.getBigInt64(i * 8, true))

    return labels
  }

  /**
   * Reads fixed-width UCS4 strings, as NumPy stores text arrays.
   *
   * @param array The parsed class_names member.
   * @returns The decoded strings, with padding NULs removed.
   * @throws If the member is not little-endian UCS4.
   */
  private static parseStrings(array: NpyArray): string[] {
    const match = array.descr.match(/^<[<>]?U(\d+)$/) ?? array.descr.match(/^U(\d+)$/)
    if (match === null) throw new Error(`Expected a UCS4 string array, got dtype ${array.descr}`)

    const charactersPerItem = Number(match[1])
    const stride = charactersPerItem * 4
    const strings: string[] = []

    for (let i = 0; i < array.shape[0]; i++) {
      let text = ''

      for (let c = 0; c < charactersPerItem; c++) {
        const codePoint = array.data.getUint32(i * stride + c * 4, true)
        if (codePoint === 0) continue

        text += String.fromCodePoint(codePoint)
      }

      strings.push(text)
    }

    return strings
  }

  /**
   * Builds a deterministic permutation of [0, count).
   *
   * The bundle stores images grouped by class, so a sequential split would put
   * whole classes in only one split. Shuffling first interleaves them.
   *
   * @param count How many indices to permute.
   * @param seed Seed for the xorshift generator, so splits are reproducible.
   * @returns The permuted indices.
   */
  private static shuffledIndices(count: number, seed: number): number[] {
    const indices = Array.from({ length: count }, (_, i) => i)
    let state = seed >>> 0 || 1

    for (let i = count - 1; i > 0; i--) {
      // xorshift32: cheap, deterministic and good enough to interleave classes.
      state ^= state << 13
      state ^= state >>> 17
      state ^= state << 5
      state >>>= 0

      const j = state % (i + 1)
      const swap = indices[i]

      indices[i] = indices[j]
      indices[j] = swap
    }

    return indices
  }
}