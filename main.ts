import { NETWORK_LAYER_CONFIG, TRAINING_CONFIG } from './constants.ts'
import { AnimalMnistLoader } from './animal_mnist_loader.ts'
import { MnistLoader } from './mnist_loader.ts'
import {
  DATASET_FLAG_USAGE,
  parseDatasetFlag,
  wantsHelp,
} from './dataset.ts'
import type { DatasetChoice, DatasetLoader } from './dataset.ts'
import { Network } from './network.ts'
import {
  assertIsNotNull,
  timesMap,
  shuffle,
  formatPercentageWithDecimalPlaces,
} from './utils.ts'
import { Gradient } from './types.ts'
import { timesForEach } from './utils.ts'

// The network and loader are kept at module scope (rather than local to `main`) so
// they stay accessible (and inspectable) from the Deno console after training.
let network: Network
let loader: DatasetLoader

/**
 * Builds the loader for the chosen dataset. Both loaders expose the same
 * {@link DatasetLoader} surface, so the rest of `main` is dataset-agnostic.
 *
 * @param choice Which dataset to read.
 * @returns A loader for that dataset, not yet loaded.
 */
const createLoader = (choice: DatasetChoice): DatasetLoader =>
  choice === 'animal' ? new AnimalMnistLoader() : new MnistLoader()

/**
 * Entry point: loads the requested dataset (digits by default, animals with
 * -a), builds the network from
 * {@link NETWORK_LAYER_CONFIG}, trains it with mini-batch gradient descent for
 * as many epochs as {@link TRAINING_CONFIG} sets for that dataset, then scores
 * it on the test split.
 *
 * Each epoch reshuffles the samples and walks whole batches of
 * {@link TRAINING_CONFIG}'s `batchSize`: the gradients of the batch are
 * collected, averaged and applied in one step, so the numbers logged for a batch
 * describe the weights as they were *before* that step.
 */
const main = async () => {
  const args = Deno.args

  if (wantsHelp(args)) {
    console.log(DATASET_FLAG_USAGE)
    return
  }

  const choice = parseDatasetFlag(args)

  // Digits come from the gzipped IDX files in ./data/mnist, animals from the
  // .npz bundle in ./data/animal-mnist.
  loader = createLoader(choice)
  await loader.load(console.log)
  assertIsNotNull(loader.trainingData)
  assertIsNotNull(loader.testData)

  // Each dataset brings its own epoch count, learning rate (η) and batch size.
  const { epochs, learningRate, batchSize } = TRAINING_CONFIG[choice]

  // Build the network topology, learning rate (η) and batch size.
  network = new Network(NETWORK_LAYER_CONFIG, learningRate, batchSize)

  console.log({ dataset: choice, epochs, learningRate, batchSize })
  console.log({ parameterCount: network.parameterCount })

  const { inputs, labels } = loader.trainingData
  const totalBatches = Math.trunc(inputs.length / batchSize)
  const trainingDataIndexes = timesMap(inputs.length, i => i)

  let avgCost = 0
  let hits: number
  let misses: number

  timesForEach(epochs, epochIdx => {
    hits = 0
    misses = 0

    shuffle(trainingDataIndexes)

    timesForEach(totalBatches, batchIdx => {
      const batchGradients: Gradient[] = []

      timesForEach(batchSize, batchImageIdx => {
        const sampleIndex =
          trainingDataIndexes[batchIdx * batchSize + batchImageIdx]

        network.loadSample(inputs[sampleIndex], labels[sampleIndex])
        network.feedForward()

        const predictedDigit = network.predictedDigit()

        if (predictedDigit === labels[sampleIndex]) hits++
        else misses++

        batchGradients.push(network.calculateGradient())
      })

      const averageGradient = network.calculateAverageGradient(batchGradients)

      network.backPropagate(averageGradient)

      avgCost =
        ((epochIdx * totalBatches + batchIdx) * avgCost + network.cost) /
        (epochIdx * totalBatches + (batchIdx + 1))

      console.log({
        epoch: `${epochIdx + 1}/${epochs}`,
        batch: `${batchIdx + 1}/${totalBatches}`,
        cost: network.cost,
        hits,
        misses,
        avgCost,
        epochAccuracy: `${formatPercentageWithDecimalPlaces(hits / ((batchIdx + 1) * batchSize), 2)}%`,
      })
    })
  })

  console.log('-----------------------------------------')
  console.log('Checking network accuracy with test data:')
  console.log('-----------------------------------------')

  const { inputs: testInputs, labels: testLabels } = loader.testData

  hits = 0
  misses = 0

  timesForEach(testInputs.length, testImageIdx => {
    network.loadSample(testInputs[testImageIdx], testLabels[testImageIdx])
    network.feedForward()

    const predictedDigit = network.predictedDigit()

    if (predictedDigit === testLabels[testImageIdx]) hits++
    else misses++

    if ((testImageIdx + 1) % 1000 === 0)
      console.log({
        i: testImageIdx,
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
    dataset: choice,
    epochs,
    batchSize,
    η: network.η,
    cost: network.cost,
    hits,
    misses,
    finalAccuracy: `${formatPercentageWithDecimalPlaces(hits / testInputs.length, 2)}%`,
  })
}

await main()
