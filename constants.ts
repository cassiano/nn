// Central place for project-wide constants shared across modules.
//
// Dataset constants (MNIST magic numbers, image dimensions, file paths) live
// alongside their loader in mnist_loader.ts. Add any other global numeric or
// string constants here as the project grows.

import { InitialLayerData } from './types.ts'

// 784 pixels                           784 x 16 = 12544 weights
//   -> 16 (sigmoid)   -> 16 (relu)     16  x 16 =   256 weights
//     -> 10 (softmax)                  16  x 10 =   160 weights
//
// 12960 weights + 42 biases (for a total of 13002 parameters).
export const NETWORK_LAYER_CONFIG: InitialLayerData[] = [
  { name: 'Input Layer', size: 784 }, // MNIST_IMAGE_COLS * MNIST_IMAGE_ROWS
  { name: 'Hidden Layer 1', size: 16, σ: 'relu' },
  { name: 'Hidden Layer 2', size: 16, σ: 'relu' },
  { name: 'Output Layer', size: 10, σ: 'softmax' }, // One neuron per digit (0-9)
]

// η = learning rate
export const NETWORK_LEARNING_RATE = 0.002

export const EPOCHS = 5
export const BATCH_SIZE = 60
