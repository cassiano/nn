import { MnistLoader } from './mnist_loader.ts'
import { Network } from './network.ts'

const main = async () => {
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

  // Get 1st training sample.
  network.setInputs(
    loader.getTrainingData().inputs[0],
    loader.getTrainingData().labels[0],
  )

  network.forward()

  console.log(network.cost())
}

await main()
