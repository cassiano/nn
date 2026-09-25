import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'
import { assertIsNotNull } from './utils.ts'

// The network is kept at module scope (rather than local to `main`) so it
// stays accessible (and inspectable) from the Deno console after training.
let network: Network
let loader: MnistLoader

const main = async () => {
  // Running mean of the network cost over every sample processed so far.
  // Updated incrementally so we don't need to re-sum all previous losses.
  let averageNetworkCost = 0

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

    averageNetworkCost = (averageNetworkCost * i + network.cost) / (i + 1)

    // Log progress every 1000 samples so long runs stay observable.
    if ((i + 1) % 1000 === 0)
      console.log({ count: i + 1, averageNetworkCost: averageNetworkCost })
  }

  console.log({ finalAverageNetworkCost: averageNetworkCost })
}

await main()
