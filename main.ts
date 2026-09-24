import { Layer } from './layer.ts'
import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'

const main = async () => {
  let averageNetworkCost = 0

  const loader = new MnistLoader()
  await loader.load(console.log)

  const network = new Network([
    new Layer('Input Layer', 784),
    new Layer('Hidden Layer 1', 16, 'sigmoid'),
    new Layer('Hidden Layer 2', 16, 'relu'),
    new Layer('Output Layer', 10, 'softmax'),
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
