import { NETWORK_LAYER_CONFIG, TRAINING_CONFIG } from '../constants.ts'
import {
  DATASET_IMAGE_COLS,
  DATASET_IMAGE_ROWS,
  DATASET_OUTPUT_SIZE,
  parseDatasetFlag,
} from '../dataset.ts'
import { assert, assertEquals } from './test_helpers.ts'

Deno.test('constants / the architecture matches the shared dataset geometry', () => {
  const [input, ...rest] = NETWORK_LAYER_CONFIG
  const output = rest[rest.length - 1]

  assertEquals(input.size, DATASET_IMAGE_ROWS * DATASET_IMAGE_COLS)
  assertEquals(output.size, DATASET_OUTPUT_SIZE)
})

Deno.test('constants / every dataset has usable training settings', () => {
  for (const choice of ['digits', 'animal'] as const) {
    const config = TRAINING_CONFIG[choice]

    assert(config !== undefined, `${choice} should have a training config`)
    assert(
      Number.isInteger(config.epochs) && config.epochs > 0,
      `${choice}.epochs should be a positive integer, got ${config.epochs}`,
    )
    assert(
      Number.isFinite(config.learningRate) && config.learningRate > 0,
      `${choice}.learningRate should be positive, got ${config.learningRate}`,
    )
    assert(
      Number.isInteger(config.batchSize) && config.batchSize > 0,
      `${choice}.batchSize should be a positive integer, got ${config.batchSize}`,
    )
  }
})

Deno.test('constants / the training config covers every CLI choice', () => {
  // Guards against a flag being added to dataset.ts without a matching entry.
  for (const args of [[], ['-d'], ['-a']]) {
    const choice = parseDatasetFlag(args)
    assert(
      TRAINING_CONFIG[choice] !== undefined,
      `${choice} needs an entry in TRAINING_CONFIG`,
    )
  }
})