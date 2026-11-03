import {
  NMIST_BATCH_SIZE,
  NMIST_LAYER_CONFIG,
  NMIST_LEARNING_RATE,
  NMIST_EPOCHS,
} from './constants.ts'
import { MnistLoader } from './mnist_loader.ts'
import {
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
  MNIST_OUTPUT_SIZE,
} from './mnist_loader.ts'
import { Network } from './network.ts'
import {
  assertIsNotNull,
  timesMap,
  shuffle,
  formatPercentageWithDecimalPlaces,
} from './utils.ts'
import { timesForEach } from './utils.ts'

// The network and loader are kept at module scope (rather than local to `main`) so
// they stay accessible (and inspectable) from the Deno console after training.
let network: Network
let loader: MnistLoader

/** File holding the network's weights and biases, under the project's data/ directory. */
const MNIST_NETWORK_FILENAME = 'mnist_network.json'

/**
 * Command-line flag that makes the network start from the weights and biases
 * saved in {@link MNIST_NETWORK_FILENAME} instead of fresh random ones.
 */
const LOAD_INITIAL_PARAMETERS_FLAG = '-l'

/**
 * Command-line flag that skips training entirely and goes straight to scoring
 * the test split, with or without {@link LOAD_INITIAL_PARAMETERS_FLAG}.
 */
const SKIP_TRAINING_FLAG = '-s'

/**
 * Entry point: loads MNIST, builds the network from
 * {@link NMIST_LAYER_CONFIG}, trains it with mini-batch gradient descent for
 * {@link NMIST_EPOCHS} epochs, then scores it on the test split.
 *
 * Pass `-l` to start from the weights and biases in
 * {@link MNIST_NETWORK_FILENAME}; without it the network is randomly
 * initialized as usual. Pass `-s` to skip training and score the test split
 * right away, using either the random parameters or the ones loaded with `-l`.
 *
 * Each epoch reshuffles the samples and walks whole batches of
 * {@link NMIST_BATCH_SIZE}: the gradients of the batch are collected, averaged and
 * applied in one step, so the numbers logged for a batch describe the weights
 * as they were *before* that step.
 */
const main = async () => {
  const args = new Set(Deno.args)
  const shouldLoadInitialParameters = args.has(LOAD_INITIAL_PARAMETERS_FLAG)
  const shouldSkipTraining = args.has(SKIP_TRAINING_FLAG)

  // Load the MNIST dataset from the zipped IDX files in the project's
  // data/mnist directory.
  loader = new MnistLoader()
  await loader.load(console.log)
  assertIsNotNull(loader.trainingData)
  assertIsNotNull(loader.testData)

  // Build the network topology and learning rate (η).
  network = new Network(NMIST_LAYER_CONFIG, NMIST_LEARNING_RATE, {
    inputSize: MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS,
    outputSize: MNIST_OUTPUT_SIZE,
    batchSize: NMIST_BATCH_SIZE,
  })

  if (shouldLoadInitialParameters) {
    await network.loadFromDisk(MNIST_NETWORK_FILENAME)
    console.log(`Loaded initial parameters from ${MNIST_NETWORK_FILENAME}`)
  }

  console.log({ parameterCount: network.parameterCount })

  let hits: number
  let misses: number

  if (shouldSkipTraining) {
    console.log('Skipping training')
  } else {
    const { inputs, labels } = loader.trainingData
    const totalBatches = Math.trunc(inputs.length / NMIST_BATCH_SIZE)
    const trainingDataIndexes = timesMap(inputs.length, i => i)

    let avgCost = 0

    timesForEach(NMIST_EPOCHS, epochIdx => {
      hits = 0
      misses = 0

      shuffle(trainingDataIndexes)

      timesForEach(totalBatches, batchIdx => {
        const batchGradients = timesMap(NMIST_BATCH_SIZE, batchImageIdx => {
          const sampleIndex =
            trainingDataIndexes[batchIdx * NMIST_BATCH_SIZE + batchImageIdx]

          network.loadSample(inputs[sampleIndex], labels[sampleIndex])
          network.feedForward()

          const predicted = network.predictedClass()

          if (predicted === labels[sampleIndex]) hits++
          else misses++

          return network.calculateGradient()
        })

        network.backPropagate(network.calculateAverageGradient(batchGradients))

        avgCost =
          ((epochIdx * totalBatches + batchIdx) * avgCost + network.cost) /
          (epochIdx * totalBatches + (batchIdx + 1))

        console.log({
          epoch: `${epochIdx + 1}/${NMIST_EPOCHS}`,
          batch: `${batchIdx + 1}/${totalBatches}`,
          cost: network.cost,
          hits,
          misses,
          avgCost,
          epochAccuracy: `${formatPercentageWithDecimalPlaces(hits / ((batchIdx + 1) * NMIST_BATCH_SIZE), 2)}%`,
        })
      })
    })

    network.saveToDisk(MNIST_NETWORK_FILENAME)
  }

  console.log('-----------------------------------------')
  console.log('Checking network accuracy with test data:')
  console.log('-----------------------------------------')

  const { inputs: testInputs, labels: testLabels } = loader.testData

  hits = 0
  misses = 0

  timesForEach(testInputs.length, testImageIdx => {
    network.loadSample(testInputs[testImageIdx], testLabels[testImageIdx])
    network.feedForward()

    const predictedDigit = network.predictedClass()

    if (predictedDigit === testLabels[testImageIdx]) hits++
    else misses++

    if ((testImageIdx + 1) % 1000 === 0)
      console.log({
        testImageIdx,
        expected: testLabels[testImageIdx],
        predictedDigit,
        cost: network.cost,
        hits,
        misses,
        effectiveAccuracy: `${formatPercentageWithDecimalPlaces(hits / (testImageIdx + 1), 2)}%`,
      })
  })

  console.log('--------------')
  console.log('Final results:')
  console.log('--------------')

  console.log({
    epochs: shouldSkipTraining ? 0 : NMIST_EPOCHS,
    η: network.η,
    cost: network.cost,
    hits,
    misses,
    finalAccuracy: `${formatPercentageWithDecimalPlaces(hits / testInputs.length, 2)}%`,
  })
}

if (import.meta.main) await main()
