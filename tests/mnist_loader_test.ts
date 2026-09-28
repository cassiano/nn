import {
  MnistLoader,
  MNIST_OUTPUT_SIZE,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
  MNIST_IMAGE_MAGIC,
  MNIST_LABEL_MAGIC,
  MNIST_PIXEL_MAX,
} from '../mnist_loader.ts'
import { assert, assertEquals, assertThrows } from './test_helpers.ts'

Deno.test(
  'MNIST / dataset constants describe the canonical dataset layout',
  () => {
    assertEquals(MNIST_OUTPUT_SIZE, 10) // digits 0-9
    assertEquals(MNIST_IMAGE_ROWS, 28)
    assertEquals(MNIST_IMAGE_COLS, 28)
    assertEquals(MNIST_IMAGE_MAGIC, 2051) // 0x00000803
    assertEquals(MNIST_LABEL_MAGIC, 2049) // 0x00000801
    assertEquals(MNIST_PIXEL_MAX, 255) // pixel byte max
  },
)

Deno.test('MnistLoader / imageToText renders a blank image as spaces', () => {
  const text = MnistLoader.imageToText(new Array(784).fill(0))
  const rows = text.split('\n').slice(0, MNIST_IMAGE_ROWS)
  assertEquals(rows.length, MNIST_IMAGE_ROWS)
  for (const row of rows) {
    // Characters are doubled horizontally for near-square pixels.
    assertEquals(row.length, MNIST_IMAGE_COLS * 2)
    assertEquals(row, ' '.repeat(MNIST_IMAGE_COLS * 2))
  }
})

Deno.test(
  'MnistLoader / imageToText renders a white image as full blocks',
  () => {
    const rows = MnistLoader.imageToText(new Array(784).fill(1))
      .split('\n')
      .slice(0, MNIST_IMAGE_ROWS)
    for (const row of rows) assertEquals(row, '█'.repeat(MNIST_IMAGE_COLS * 2))
  },
)

Deno.test(
  'MnistLoader / imageToText maps brightness onto the character gradient',
  () => {
    // gradient = ' ░▒▓▉█'; map clamps into [0, 5] then truncates.
    const pixel = (value: number) =>
      MnistLoader.imageToText(
        Array.from({ length: 784 }, (_, i) => (i === 0 ? value : 0)),
      ).split('\n')[0][0]

    assertEquals(pixel(0), ' ') // value 0 -> index 0 (space)
    assertEquals(pixel(0.5), '▒') // value 0.5 -> index 2
    assertEquals(pixel(0.6), '▓') // value 0.6 -> index 3
    assertEquals(pixel(1), '█') // value 1 -> clamped to index 5
  },
)

Deno.test('MnistLoader / imageAsText throws before load()', () => {
  const loader = new MnistLoader()
  assertThrows(() => loader.imageAsText('trainingData', 0), 'not to be null')
})

Deno.test(
  'MnistLoader / load parses the bundled dataset end-to-end',
  async () => {
    const loader = new MnistLoader()
    await loader.load()
    assert(loader.loaded)
    assert(loader.trainingData !== null)
    assert(loader.testData !== null)

    // Canonical MNIST split sizes.
    assertEquals(loader.trainingData.inputs.length, 60_000)
    assertEquals(loader.trainingData.labels.length, 60_000)
    assertEquals(loader.testData.inputs.length, 10_000)
    assertEquals(loader.testData.labels.length, 10_000)

    // Every image is a flattened 28x28 vector of pixels normalized to [0, 1].
    for (const image of loader.trainingData.inputs.slice(0, 50)) {
      assertEquals(image.length, MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
      for (const pixel of image) assert(pixel >= 0 && pixel <= 1)
    }

    // Labels are integers in [0, 9].
    for (const label of loader.trainingData.labels.slice(0, 1000)) {
      assert(Number.isInteger(label) && label >= 0 && label <= 9)
    }
  },
)

Deno.test('MnistLoader / load is idempotent', async () => {
  const loader = new MnistLoader()
  await loader.load()
  const inputs = loader.trainingData?.inputs
  await loader.load()
  assertEquals(loader.trainingData?.inputs, inputs)
  assertEquals(loader.loaded, true)
})

Deno.test(
  'MnistLoader / imageAsText renders dataset images after load()',
  async () => {
    const loader = new MnistLoader()
    await loader.load()
    const rows = loader
      .imageAsText('trainingData', 0)
      .split('\n')
      .slice(0, MNIST_IMAGE_ROWS)
    assertEquals(rows.length, MNIST_IMAGE_ROWS)
    for (const row of rows) assertEquals(row.length, MNIST_IMAGE_COLS * 2)
  },
)
