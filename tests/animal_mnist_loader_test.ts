import {
  AnimalMnistLoader,
  ANIMAL_CLASS_NAMES,
  ANIMAL_SAMPLE_COUNT,
  ANIMAL_DEFAULT_TEST_FRACTION,
} from '../animal_mnist_loader.ts'
import {
  DATASET_OUTPUT_SIZE,
  DATASET_IMAGE_COLS,
  DATASET_IMAGE_ROWS,
  DATASET_PIXEL_MAX,
} from '../dataset.ts'
import { assert, assertEquals, assertThrows } from './test_helpers.ts'

Deno.test(
  'Animal-MNIST / dataset constants describe the canonical dataset layout',
  () => {
    assertEquals(DATASET_OUTPUT_SIZE, 10) // ten animal classes
    assertEquals(DATASET_IMAGE_ROWS, 28)
    assertEquals(DATASET_IMAGE_COLS, 28)
    assertEquals(DATASET_PIXEL_MAX, 255) // pixel byte max
    assertEquals(ANIMAL_SAMPLE_COUNT, 10_000)
    assertEquals(ANIMAL_CLASS_NAMES.length, DATASET_OUTPUT_SIZE)
    assertEquals(ANIMAL_DEFAULT_TEST_FRACTION, 0.2)
  },
)

Deno.test('AnimalMnistLoader / imageToText renders a blank image as spaces', () => {
  const text = AnimalMnistLoader.imageToText(new Array(784).fill(0))
  const rows = text.split('\n').slice(0, DATASET_IMAGE_ROWS)
  assertEquals(rows.length, DATASET_IMAGE_ROWS)
  for (const row of rows) {
    // Characters are doubled horizontally for near-square pixels.
    assertEquals(row.length, DATASET_IMAGE_COLS * 2)
    assertEquals(row, ' '.repeat(DATASET_IMAGE_COLS * 2))
  }
})

Deno.test(
  'AnimalMnistLoader / imageToText renders a white image as full blocks',
  () => {
    const rows = AnimalMnistLoader.imageToText(new Array(784).fill(1))
      .split('\n')
      .slice(0, DATASET_IMAGE_ROWS)
    for (const row of rows) assertEquals(row, '█'.repeat(DATASET_IMAGE_COLS * 2))
  },
)

Deno.test(
  'AnimalMnistLoader / imageToText maps brightness onto the character gradient',
  () => {
    // gradient = ' ░▒▓▉█'; map clamps into [0, 5] then truncates.
    const pixel = (value: number) =>
      AnimalMnistLoader.imageToText(
        Array.from({ length: 784 }, (_, i) => (i === 0 ? value : 0)),
      ).split('\n')[0][0]

    assertEquals(pixel(0), ' ') // value 0 -> index 0 (space)
    assertEquals(pixel(0.5), '▒') // value 0.5 -> index 2
    assertEquals(pixel(0.6), '▓') // value 0.6 -> index 3
    assertEquals(pixel(1), '█') // value 1 -> clamped to index 5
  },
)

Deno.test('AnimalMnistLoader / imageAsText throws before load()', () => {
  const loader = new AnimalMnistLoader()
  assertThrows(() => loader.imageAsText('trainingData', 0), 'not to be null')
})

Deno.test(
  'AnimalMnistLoader / load parses the bundled dataset end-to-end',
  async () => {
    const loader = new AnimalMnistLoader()
    await loader.load()
    assert(loader.loaded)
    assert(loader.trainingData !== null)
    assert(loader.testData !== null)

    // The bundle holds 10,000 images, split 80/20 by default.
    const testCount = Math.round(
      ANIMAL_SAMPLE_COUNT * ANIMAL_DEFAULT_TEST_FRACTION,
    )
    assertEquals(
      loader.trainingData.inputs.length,
      ANIMAL_SAMPLE_COUNT - testCount,
    )
    assertEquals(loader.trainingData.labels.length, ANIMAL_SAMPLE_COUNT - testCount)
    assertEquals(loader.testData.inputs.length, testCount)
    assertEquals(loader.testData.labels.length, testCount)

    // Every image is a flattened 28x28 vector of pixels normalized to [0, 1].
    for (const image of loader.trainingData.inputs.slice(0, 50)) {
      assertEquals(image.length, DATASET_IMAGE_ROWS * DATASET_IMAGE_COLS)
      for (const pixel of image) assert(pixel >= 0 && pixel <= 1)
    }

    // Labels are integers in [0, 9].
    for (const label of loader.trainingData.labels.slice(0, 1000)) {
      assert(Number.isInteger(label) && label >= 0 && label <= 9)
    }
  },
)

Deno.test('AnimalMnistLoader / load reads class names from the bundle', async () => {
  const loader = new AnimalMnistLoader()
  await loader.load()

  assert(loader.classNames !== null)
  assertEquals(loader.classNames.length, DATASET_OUTPUT_SIZE)
  assertEquals(loader.classNames, [...ANIMAL_CLASS_NAMES])
  assertEquals(loader.className(0), 'Bear')
  assertEquals(loader.className(9), 'Zebra')
})

Deno.test(
  'AnimalMnistLoader / splitting keeps every class in both splits',
  async () => {
    // The bundle groups images by class, so an unshuffled split would leave
    // whole classes in only one split. Both must stay fully stratified.
    const loader = new AnimalMnistLoader()
    await loader.load()

    const present = (labels: number[]) =>
      new Set(labels).size === DATASET_OUTPUT_SIZE

    assert(present(loader.trainingData!.labels))
    assert(present(loader.testData!.labels))
  },
)

Deno.test(
  'AnimalMnistLoader / splitting is reproducible across loaders',
  async () => {
    const first = new AnimalMnistLoader()
    const second = new AnimalMnistLoader()
    await first.load()
    await second.load()

    assertEquals(first.trainingData!.labels, second.trainingData!.labels)
    assertEquals(first.testData!.inputs[0], second.testData!.inputs[0])
  },
)

Deno.test(
  'AnimalMnistLoader / splits partition the dataset and preserve class balance',
  async () => {
    const loader = new AnimalMnistLoader()
    await loader.load()
    const training = loader.trainingData!
    const test = loader.testData!

    // Together the splits hold every sample exactly once.
    assertEquals(training.inputs.length + test.inputs.length, ANIMAL_SAMPLE_COUNT)

    // The dataset is balanced, so each class totals 1,000 across both splits.
    // Note the bundle contains 284 duplicated image pairs, so images themselves
    // are not unique and cannot be used to prove the partition.
    const totals = new Map<number, number>()
    for (const label of [...training.labels, ...test.labels])
      totals.set(label, (totals.get(label) ?? 0) + 1)

    assertEquals(totals.size, DATASET_OUTPUT_SIZE)
    for (const [label, count] of totals)
      assertEquals(
        count,
        ANIMAL_SAMPLE_COUNT / DATASET_OUTPUT_SIZE,
        `class ${label} should hold ${ANIMAL_SAMPLE_COUNT / DATASET_OUTPUT_SIZE} samples`,
      )
  },
)

Deno.test('AnimalMnistLoader / load is idempotent', async () => {
  const loader = new AnimalMnistLoader()
  await loader.load()
  const inputs = loader.trainingData?.inputs
  await loader.load()
  assertEquals(loader.trainingData?.inputs, inputs)
  assertEquals(loader.loaded, true)
})

Deno.test(
  'AnimalMnistLoader / imageAsText renders dataset images after load()',
  async () => {
    const loader = new AnimalMnistLoader()
    await loader.load()
    const rows = loader
      .imageAsText('trainingData', 0)
      .split('\n')
      .slice(0, DATASET_IMAGE_ROWS)
    assertEquals(rows.length, DATASET_IMAGE_ROWS)
    for (const row of rows) assertEquals(row.length, DATASET_IMAGE_COLS * 2)
  },
)

Deno.test('AnimalMnistLoader / load throws when the bundle is missing', async () => {
  const loader = new AnimalMnistLoader({ path: './data/animal-mnist/absent.npz' })
  let thrown: unknown = null
  try {
    await loader.load()
  } catch (error) {
    thrown = error
  }
  assert(thrown instanceof Deno.errors.NotFound)
  assertEquals(loader.loaded, false)
})