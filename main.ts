import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'
import { assertIsNotNull } from './utils.ts'

let network: Network

const main = async () => {
  let averageNetworkCost = 0

  const loader = new MnistLoader()
  await loader.load(console.log)
  assertIsNotNull(loader.trainData)

  network = new Network(
    [
      { name: 'Input Layer', size: 784 },
      { name: 'Hidden Layer 1', size: 16, σ: 'sigmoid' },
      { name: 'Hidden Layer 2', size: 16, σ: 'relu' },
      { name: 'Output Layer', size: 10, σ: 'softmax' },
    ],
    0.01,
  )

  console.log({ parameterCount: network.parameterCount })

  const images = loader.trainData.inputs
  const labels = loader.trainData.labels

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
