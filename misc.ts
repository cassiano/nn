import { MnistLoader } from './mnist_loader.ts'
import { Network } from './types.ts'
import { fromMatrix, addMatrices } from './utils.ts'
import { sigmoid, relu, tanh } from './activation.ts'
import {
  timesMap,
  random,
  multiplyMatrices,
  assertIsNotUndefined,
  toMatrix,
} from './utils.ts'
import { softmax } from './activation.ts'

// C = ∑(y - a(𝐋))²
const cost = (network: Network): number => {
  const outputLayer = network.layers[network.layers.length - 1]

  return timesMap(
    outputLayer.a.length,
    i => (network.y[i] - outputLayer.a[i]) ** 2,
  ).reduce((acc, item) => acc + item)
}

const network: Network = {
  layers: [
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
  ],
  y: [],
}

for (let i = 1; i < network.layers.length; i++) {
  network.layers[i].w = timesMap(network.layers[i].size, () =>
    timesMap(network.layers[i - 1].size, () => random(-1, 1)),
  )

  network.layers[i].b = timesMap(network.layers[i].size, () => random(-1, 1))
}

const loader = new MnistLoader()
await loader.load(console.log)

// Get 1st training sample and set it as the activation of the input layer
network.layers[0].a = loader.getTrainingData().inputs[0]

network.y = timesMap(10, i =>
  i === loader.getTrainingData().labels[0] ? 1 : 0,
)

for (let i = 1; i < network.layers.length; i++) {
  console.log(`Processing layer ${i}: ${network.layers[i].name}`)

  const currentLayer = network.layers[i]
  const previousLayer = network.layers[i - 1]

  assertIsNotUndefined(currentLayer.σ)
  assertIsNotUndefined(currentLayer.w)
  assertIsNotUndefined(currentLayer.b)

  // z(𝓁) = w(𝓁) * a(𝓁-1) + b(𝓁)
  currentLayer.z = fromMatrix(
    addMatrices(
      multiplyMatrices(currentLayer.w, toMatrix(previousLayer.a)),
      toMatrix(currentLayer.b),
    ),
  )

  switch (currentLayer.σ) {
    case 'sigmoid':
      currentLayer.a = currentLayer.z.map(sigmoid)
      break
    case 'relu':
      currentLayer.a = currentLayer.z.map(relu)
      break
    case 'tanh':
      currentLayer.a = currentLayer.z.map(tanh)
      break
    case 'softmax':
      currentLayer.a = softmax(currentLayer.z)
      break
    default: {
      const exhaustiveCheck: never = currentLayer.σ
      throw exhaustiveCheck
    }
  }
}

console.log(cost(network))
