import {
  BATCH_SIZE,
  NETWORK_LAYER_CONFIG,
  NETWORK_LEARNING_RATE,
  EPOCHS,
} from './constants.ts'
import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'
import { assertIsNotNull } from './utils.ts'
import { Gradient } from './types.ts'

// The network and loader are kept at module scope (rather than local to `main`) so
// they stay accessible (and inspectable) from the Deno console after training.
let network: Network
let loader: MnistLoader

/**
 * Entry point: loads MNIST, builds the 784 -> 16 -> 16 -> 10 network, then runs
 * a single training pass over every image.
 *
 * For each sample it loads the input, runs {@link Network.feedForward}, logs
 * the prediction alongside the true label and the cost, then calls
 * {@link Network.backPropagate} to step the weights. One pass is one epoch of
 * stochastic gradient descent; there is no shuffling or batching yet.
 *
 * @returns Nothing. Resolves once every training sample has been processed.
 * @throws If the dataset files are missing or malformed, or if the topology
 * fails the MNIST size validation.
 */
const main = async () => {
  // Load the MNIST dataset from the gzipped IDX files in ./mnist.
  loader = new MnistLoader()
  await loader.load(console.log)
  assertIsNotNull(loader.trainData)
  assertIsNotNull(loader.testData)

  // Build the network topology and learning rate (η).
  network = new Network(NETWORK_LAYER_CONFIG, NETWORK_LEARNING_RATE)

  console.log({ parameterCount: network.parameterCount })

  const { inputs, labels } = loader.trainData
  const totalBatches = Math.trunc(inputs.length / BATCH_SIZE)

  for (let i = 0; i < EPOCHS; i++) {
    console.log(`Starting epoch ${(i = 1)}`)

    for (let j = 0; j < totalBatches; j++) {
      console.log(`Processing batch ${j + 1} of ${totalBatches}`)

      const batchGradients: Gradient[] = []

      for (let k = 0; k < BATCH_SIZE; k++) {
        const sampleIndex = j * BATCH_SIZE + k

        network.loadSample(inputs[sampleIndex], labels[sampleIndex])
        network.feedForward()

        batchGradients.push(network.calculateGradient())
      }

      console.log('Updating network parameters')

      const averageGradient = network.calculateAverageGradient(batchGradients)

      network.backPropagate(averageGradient)
    }
  }

  console.log('-----------------------------------------')
  console.log('Checking network accuracy with test data:')
  console.log('-----------------------------------------')

  const { inputs: testInputs, labels: testLabels } = loader.testData
  let hits = 0
  let misses = 0

  for (let i = 0; i < testInputs.length; i++) {
    network.loadSample(inputs[i], labels[i])
    network.feedForward()

    const predictedDigit = network.predictedDigit()

    if (predictedDigit === testLabels[i]) hits++
    else misses++

    console.log({
      i,
      expected: testLabels[i],
      predictedDigit,
      cost: network.cost,
      hits,
      misses,
    })
  }
}

await main()
