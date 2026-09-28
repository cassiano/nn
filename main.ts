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
  formatWithDecimalPlaces,
} from './utils.ts'
import { Gradient } from './types.ts'
import { timesForEach } from './utils.ts'

// The network and loader are kept at module scope (rather than local to `main`) so
// they stay accessible (and inspectable) from the Deno console after training.
let network: Network
let loader: MnistLoader

/**
 * Entry point: loads MNIST, builds the network described by
 * {@link NETWORK_LAYER_CONFIG} (784 -> 16 -> 16 -> 10), trains it for
 * {@link EPOCHS} epochs of mini-batch gradient descent, then scores the result
 * on the test split.
 *
 * Training is a double loop over epochs and batches of {@link BATCH_SIZE}
 * samples. Each epoch shuffles a fresh index array and hands out consecutive
 * slices of it, so every epoch sees the data in a different order. Only one
 * sample is loaded into the network at a time, so each batch is accumulated
 * first: per sample {@link Network.loadSample} → {@link Network.feedForward} →
 * {@link Network.calculateGradient}, then
 * {@link Network.calculateAverageGradient} turns the batch into one mean
 * gradient and a single {@link Network.backPropagate} applies it. Only whole
 * batches are trained, so `inputs.length % BATCH_SIZE` trailing samples are
 * dropped each epoch.
 *
 * Afterwards, each test image is run through the network one at a time and the
 * prediction, cost and running hit/miss tally are logged.
 *
 * @returns Nothing. Resolves once every test sample has been scored.
 * @throws If the dataset files are missing or malformed, or if the topology
 * fails the MNIST size validation.
 */
const main = async () => {
  // Load the MNIST dataset from the gzipped IDX files in ./mnist.
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
      console.clear()
      console.log(
        `Processing batch ${batchIdx + 1}/${totalBatches} of epoch ${epochIdx + 1}`,
      )

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

      avgCost =
        ((epochIdx * totalBatches + batchIdx) * avgCost + network.cost) /
        (epochIdx * totalBatches + (batchIdx + 1))

      console.log({
        epoch: epochIdx + 1,
        batch: batchIdx + 1,
        cost: network.cost,
        hits,
        misses,
        avgCost,
        epochAccuracy: `${formatWithDecimalPlaces(hits / ((batchIdx + 1) * BATCH_SIZE), 2)}%`,
      })

      const averageGradient = network.calculateAverageGradient(batchGradients)

      network.backPropagate(averageGradient)
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
        effectiveAccuracy: `${formatWithDecimalPlaces(hits / (i + 1), 2)}%`,
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
    finalAccuracy: `${formatWithDecimalPlaces(hits / testInputs.length, 2)}%`,
  })
}

await main()
