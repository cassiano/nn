import {
  BATCH_SIZE,
  NETWORK_LAYER_CONFIG,
  NETWORK_LEARNING_RATE,
  EPOCHS,
} from './constants.ts'
import { MnistLoader } from './mnist_loader.ts'
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
let loader: MnistLoader

/**
 * Entry point: loads MNIST, builds the network from
 * {@link NETWORK_LAYER_CONFIG}, trains it with mini-batch gradient descent for
 * {@link EPOCHS} epochs, then scores it on the test split.
 *
 * Each epoch reshuffles the samples and walks whole batches of
 * {@link BATCH_SIZE}: the gradients of the batch are collected, averaged and
 * applied in one step, so the numbers logged for a batch describe the weights
 * as they were *before* that step.
 */
const main = async () => {
  // Load the MNIST dataset from the zipped IDX files in ./data/mnist.
  loader = new MnistLoader()
  await loader.load(console.log)
  assertIsNotNull(loader.trainingData)
  assertIsNotNull(loader.testData)

  // Build the network topology and learning rate (η).
  network = new Network(NETWORK_LAYER_CONFIG, NETWORK_LEARNING_RATE)

  console.log({ parameterCount: network.parameterCount })

  const { inputs, labels } = loader.trainingData
  const totalBatches = Math.trunc(inputs.length / BATCH_SIZE)
  const trainingDataIndexes = timesMap(inputs.length, i => i)

  let avgCost = 0
  let hits: number
  let misses: number

  timesForEach(EPOCHS, epochIdx => {
    hits = 0
    misses = 0

    shuffle(trainingDataIndexes)

    timesForEach(totalBatches, batchIdx => {
      const batchGradients: Gradient[] = []

      timesForEach(BATCH_SIZE, batchImageIdx => {
        const sampleIndex =
          trainingDataIndexes[batchIdx * BATCH_SIZE + batchImageIdx]

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
        epoch: `${epochIdx + 1}/${EPOCHS}`,
        batch: `${batchIdx + 1}/${totalBatches}`,
        cost: network.cost,
        hits,
        misses,
        avgCost,
        epochAccuracy: `${formatPercentageWithDecimalPlaces(hits / ((batchIdx + 1) * BATCH_SIZE), 2)}%`,
      })
    })
  })

  console.log('-----------------------------------------')
  console.log('Checking network accuracy with test data:')
  console.log('-----------------------------------------')

  const { inputs: testInputs, labels: testLabels } = loader.testData

  hits = 0
  misses = 0

  timesForEach(testInputs.length, i => {
    network.loadSample(testInputs[i], testLabels[i])
    network.feedForward()

    const predictedDigit = network.predictedDigit()

    if (predictedDigit === testLabels[i]) hits++
    else misses++

    if ((i + 1) % 1000 === 0)
      console.log({
        i,
        expected: testLabels[i],
        predictedDigit,
        cost: network.cost,
        hits,
        misses,
        effectiveAccuracy: `${formatPercentageWithDecimalPlaces(hits / (i + 1), 2)}%`,
      })
  })

  console.log('--------------')
  console.log('Final results:')
  console.log('--------------')

  console.log({
    epochs: EPOCHS,
    η: network.η,
    cost: network.cost,
    hits,
    misses,
    finalAccuracy: `${formatPercentageWithDecimalPlaces(hits / testInputs.length, 2)}%`,
  })
}

await main()
