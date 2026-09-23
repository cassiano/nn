import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'

const main = async () => {
  let averageNetworkCost = 0

  const loader = new MnistLoader()
  await loader.load(console.log)

  const network = new Network([
    {
      name: 'Input Layer',
      size: 784,
      a: [],
    },
    {
      name: 'Hidden Layer 1',
      size: 16,
      a: [],
      w: [],
      b: [],
      z: [],
      σ: 'sigmoid',
    },
    {
      name: 'Hidden Layer 2',
      size: 16,
      a: [],
      w: [],
      b: [],
      z: [],
      σ: 'relu',
    },
    {
      name: 'Output Layer',
      size: 10,
      a: [],
      w: [],
      b: [],
      z: [],
      σ: 'softmax',
    },
  ])

  console.log({ parameterCount: network.parameterCount })

  const images = loader.getTrainingData().inputs
  const labels = loader.getTrainingData().labels

  // Process all training samples.
  for (let i = 0; i < images.length; i++) {
    network.loadSample(images[i], labels[i])
    network.feedForward()

    averageNetworkCost = (averageNetworkCost * i + network.cost) / (i + 1)

    if ((i + 1) % 1000 === 0)
      console.log({ count: i + 1, averageNetworkCost: averageNetworkCost })
  }

  console.log({ finalAverageNetworkCost: averageNetworkCost })
}

await main()
