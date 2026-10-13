// Project-wide constants shared across modules. Dataset-specific values (IDX
// magic numbers, image dimensions, file paths) live next to the loader in
// mnist_loader.ts.

import { InitialLayerData } from './types.ts'

// Architecture, in feedforward order: 784 inputs, two 16-neuron hidden layers,
// 10 outputs — 12960 weights and 42 biases, 13002 parameters in total.
export const NETWORK_LAYER_CONFIG: InitialLayerData[] = [
  { name: 'Input Layer', size: 784 },
  { name: 'Hidden Layer 1', size: 16, σ: 'relu' },
  { name: 'Hidden Layer 2', size: 16, σ: 'relu' },
  { name: 'Output Layer', size: 10, σ: 'softmax' },
]

export const EPOCHS = 30
export const NETWORK_LEARNING_RATE = 0.002 // η, the learning rate
export const BATCH_SIZE = 60
