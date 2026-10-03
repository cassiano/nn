import {
  DATASET_FLAG_USAGE,
  DATASET_IMAGE_COLS,
  DATASET_IMAGE_ROWS,
  DATASET_OUTPUT_SIZE,
  DATASET_PIXEL_MAX,
  parseDatasetFlag,
  wantsHelp,
} from '../dataset.ts'
import { assert, assertEquals, assertThrows } from './test_helpers.ts'

Deno.test('dataset / constants describe the shape both loaders produce', () => {
  assertEquals(DATASET_OUTPUT_SIZE, 10)
  assertEquals(DATASET_IMAGE_ROWS, 28)
  assertEquals(DATASET_IMAGE_COLS, 28)
  assertEquals(DATASET_PIXEL_MAX, 255)
  assertEquals(DATASET_IMAGE_ROWS * DATASET_IMAGE_COLS, 784)
})

Deno.test('parseDatasetFlag / defaults to digits when given no arguments', () => {
  assertEquals(parseDatasetFlag([]), 'digits')
})

Deno.test('parseDatasetFlag / reads the animal flag', () => {
  assertEquals(parseDatasetFlag(['-a']), 'animal')
  assertEquals(parseDatasetFlag(['--animal']), 'animal')
})

Deno.test('parseDatasetFlag / reads the digits flag', () => {
  assertEquals(parseDatasetFlag(['-d']), 'digits')
  assertEquals(parseDatasetFlag(['--digits']), 'digits')
  assertEquals(parseDatasetFlag(['--mnist']), 'digits')
})

Deno.test('parseDatasetFlag / throws on an unrecognized argument', () => {
  assertThrows(() => parseDatasetFlag(['-x']), 'Unknown argument')
  assertThrows(() => parseDatasetFlag(['--digits', '-z']), 'Unknown argument')
})

Deno.test('wantsHelp / detects the help flags', () => {
  assert(wantsHelp(['-h']))
  assert(wantsHelp(['--help']))
  assert(wantsHelp(['-a', '--help']))
  assert(!wantsHelp([]))
  assert(!wantsHelp(['-a']))
})

Deno.test('dataset / usage text documents both flags', () => {
  for (const fragment of ['-a', '-d', '--animal', '--digits', '-h'])
    assert(DATASET_FLAG_USAGE.includes(fragment), `usage should mention ${fragment}`)
})