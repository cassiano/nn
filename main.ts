import { NETWORK_LAYER_CONFIG, NETWORK_LEARNING_RATE } from './constants.ts'
import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'
import { assertIsNotNull } from './utils.ts'

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

  // Build the network topology and learning rate (η).
  network = new Network(NETWORK_LAYER_CONFIG, NETWORK_LEARNING_RATE)

  console.log({ parameterCount: network.parameterCount })

  const { inputs, labels } = loader.trainData

  let hits = 0

  // Single training pass: run a forward pass for every training image and
  // track the running average of mean squared error (MSE) as a loss metric.
  for (let i = 0; i < inputs.length; i++) {
    network.loadSample(inputs[i], labels[i])
    network.feedForward()

    const predictedDigit = network.predictedDigit()

    if (predictedDigit === labels[i]) hits++

    console.log({
      i,
      expected: labels[i],
      predictedDigit,
      cost: network.cost,
      hits,
    })

    network.backPropagate()
  }
}

await main()
