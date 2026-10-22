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

export const EPOCHS = 100
export const NETWORK_LEARNING_RATE = 0.002 // η, the learning rate
export const BATCH_SIZE = 60

// The XOR gate: 2 inputs (each 0 or 1) and 2 outputs (the gate's result, 0 or
// 1). A hidden layer is needed because XOR is not linearly separable: no
// single straight line separates 00 and 11 from 01 and 10.
export const XOR_INPUT_SIZE = 2
export const XOR_OUTPUT_SIZE = 2

export const XOR_LAYER_CONFIG: InitialLayerData[] = [
  { name: 'Input Layer', size: XOR_INPUT_SIZE },
  { name: 'Hidden Layer', size: 4, σ: 'sigmoid' },
  { name: 'Output Layer', size: XOR_OUTPUT_SIZE, σ: 'softmax' },
]

export const XOR_LEARNING_RATE = 0.5 // η, the learning rate
export const XOR_EPOCHS = 2000
export const XOR_BATCH_SIZE = 4 // the whole gate fits in one batch
export const XOR_RESTARTS = 5 // fresh random weights, in case a run stalls
