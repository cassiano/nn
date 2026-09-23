import {
  random,
  timesMap,
  assertIsNotUndefined,
  fromMatrix,
  addMatrices,
  multiplyMatrices,
  toMatrix,
  timesMapN,
} from './utils.ts'
import { sigmoid, relu, tanh } from './activation.ts'
import { softmax } from './activation.ts'
export type ActivationFnType = 'sigmoid' | 'relu' | 'tanh' | 'softmax'

export type Layer = {
  name: string
  size: number
  w?: number[][] // Weights matrix for the layer (size x previous layer size)
  b?: number[] // Bias vector for the layer (size)
  z?: number[] // Pre-activation values (size). z(𝓁) = w(𝓁) * a(𝓁-1) + b(𝓁)
  a: number[] // Activation values (size)
  σ?: ActivationFnType
}

export class Network {
  y: number[] = [] // Output values (size)

  constructor(public layers: Layer[]) {
    // Fill weights and biases of all but the input layer with random data.
    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].w = timesMapN(
        [this.layers[i].size, layers[i - 1].size],
        () => random(-1, 1),
      )

      this.layers[i].b = timesMap(this.layers[i].size, () => random(-1, 1))
    }
  }

  setInputs(inputs: number[], label: number) {
    this.layers[0].a = inputs

    this.y = timesMap(10, i => (i === label ? 1 : 0))
  }

  forward() {
    for (let i = 1; i < this.layers.length; i++) {
      console.log(`Processing layer ${i}: ${this.layers[i].name}`)

      const currentLayer = this.layers[i]
      const previousLayer = this.layers[i - 1]

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
  }

  // C = ∑(y - a(𝐋))²
  cost(): number {
    const outputLayer = this.layers[this.layers.length - 1]

    return timesMap(
      outputLayer.a.length,
      i => (this.y[i] - outputLayer.a[i]) ** 2,
    ).reduce((acc, item) => acc + item)
  }
}
