import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'

const main = async () => {
  let networkCost = 0

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
      σ: 'sigmoid',
      a: [],
    },
    {
      name: 'Hidden Layer 2',
      size: 16,
      σ: 'relu',
      a: [],
    },
    {
      name: 'Output Layer',
      size: 10,
      σ: 'softmax',
      a: [],
    },
  ])

  const images = loader.getTrainingData().inputs
  const labels = loader.getTrainingData().labels

  // Process all training samples.
  for (let i = 0; i < images.length; i++) {
    network.setInputs(images[i], labels[i])
    network.feedForward()

    networkCost = (networkCost * i + network.cost()) / (i + 1)

    if ((i + 1) % 1000 === 0) console.log({ count: i + 1, networkCost })
  }

  console.log({ finalNetworkCost: networkCost })
}

await main()
