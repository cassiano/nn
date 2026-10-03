// Project-wide constants shared across modules. Format-specific values (IDX
// magic numbers, image dimensions, file paths) live next to their loader in
// mnist_loader.ts and animal_mnist_loader.ts, while the geometry both datasets
// share lives in dataset.ts.

import type { InitialLayerData } from './types.ts'
import type { DatasetChoice } from './dataset.ts'

// Architecture, in feedforward order: 784 inputs, two 16-neuron hidden layers,
// 10 outputs — 12960 weights and 42 biases, 13002 parameters in total. The
// topology suits both datasets, since each is 28x28 grayscale with 10 classes.
export const NETWORK_LAYER_CONFIG: InitialLayerData[] = [
  { name: 'Input Layer', size: 784 },
  { name: 'Hidden Layer 1', size: 16, σ: 'relu' },
  { name: 'Hidden Layer 2', size: 16, σ: 'relu' },
  { name: 'Output Layer', size: 10, σ: 'softmax' },
]

/**
 * The optimizer settings one dataset trains with.
 */
export type TrainingConfig = {
  /** Passes over the training split before the test split is scored. */
  epochs: number
  /** η, the learning rate scaling each gradient step. */
  learningRate: number
  /** Samples collected into one averaged gradient, and so one update. */
  batchSize: number
}

/**
 * Optimizer settings per dataset, keyed by the `-a` / `-d` choice.
 *
 * The two sets differ in size (60k digit samples vs 8k animal ones), so the
 * epoch counts, learning rates and batch sizes that suit one need not suit the
 * other. On this machine one epoch costs roughly 13s for digits and 0.9s for
 * animals, which is worth keeping in mind when raising `epochs`.
 *
 * `batchSize` also reaches the `Network`, because {@link Network.backPropagate}
 * scales a batch update by it: `w ← w - η·batchSize·mean(∂C/∂w)`. η therefore
 * stays a per-sample rate, so changing `batchSize` changes the magnitude of each
 * step, not just its variance.
 */
export const TRAINING_CONFIG: Record<DatasetChoice, TrainingConfig> = {
  digits: { epochs: 30, learningRate: 0.002, batchSize: 60 },
  animal: { epochs: 225, learningRate: 0.00275, batchSize: 8 },
}
