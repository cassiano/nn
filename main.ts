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
  // 1. Load the MNIST dataset from the gzipped IDX files in ./mnist.
  loader = new MnistLoader()
  await loader.load(console.log)
  assertIsNotNull(loader.trainData)

  // 2. Build the network topology and learning rate (η).
  //    784 pixels                           784 x 16 = 12544 weights
  //      -> 16 (sigmoid)   -> 16 (relu)     16   x 16 =   256 weights
  //        -> 10 (softmax)                  16   x 10 =   160 weights
  //    That's 12960 weights + 42 biases. Forward pass only (no backprop yet).
  network = new Network(
    [
      { name: 'Input Layer', size: 784 }, // MNIST_IMAGE_COLS * MNIST_IMAGE_ROWS
      { name: 'Hidden Layer 1', size: 16, σ: 'sigmoid' },
      { name: 'Hidden Layer 2', size: 16, σ: 'relu' },
      { name: 'Output Layer', size: 10, σ: 'softmax' }, // One neuron per digit (0-9)
    ],
    0.01, // η = learning rate
  )

  console.log({ parameterCount: network.parameterCount })

  const images = loader.trainData.inputs
  const labels = loader.trainData.labels

  // 3. Single training pass: run a forward pass for every training image and
  //    track the running average of mean squared error (MSE) as a loss metric.
  for (let i = 0; i < images.length; i++) {
    network.loadSample(images[i], labels[i])
    network.feedForward()

    console.log({
      i,
      expected: labels[i],
      predicted: network.outputLayer.a,
      cost: network.cost,
    })

    network.backPropagate()
  }
}

await main()
