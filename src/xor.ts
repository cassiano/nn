import {
  XOR_BATCH_SIZE,
  XOR_EPOCHS,
  XOR_INPUT_SIZE,
  XOR_LEARNING_RATE,
  XOR_LAYER_CONFIG,
  XOR_OUTPUT_SIZE,
  XOR_RESTARTS,
} from './constants.ts'
import { Network } from './network.ts'
import { NumericVector, TrainingData } from './types.ts'
import { formatPercentageWithDecimalPlaces, timesForEach } from './utils.ts'

/**
 * The XOR gate as a dataset: the four possible input pairs and the value the
 * gate outputs for each. The output is 1 when the inputs differ, 0 otherwise.
 */
export const XOR_DATA: TrainingData = {
  inputs: [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ],
  labels: [0, 1, 1, 0],
}

/** One row of the truth table: what went in, what came out, what was expected. */
export type XorRow = {
  /** The two inputs fed to the network. */
  inputs: NumericVector
  /** The gate's result for those inputs. */
  expected: number
  /** The class the network predicted. */
  predicted: number
  /** The output layer's activations, one per class. */
  outputs: NumericVector
}

/** The result of running a network over the whole gate. */
export type XorEvaluation = {
  /** How many rows the network got right. */
  hits: number
  /** How many rows the network got wrong. */
  misses: number
  /** Hits over the total number of rows, in [0, 1]. */
  accuracy: number
  /** One row per input pair, in dataset order. */
  rows: XorRow[]
}

/** What a training run produced. */
export type XorTrainingResult = {
  /** The trained network, with its parameters already updated. */
  network: Network
  /** The dataset the network was trained on. */
  data: TrainingData
  /** The mean cost over the dataset after the last epoch. */
  cost: number
  /** How many epochs the winning run took. */
  epochs: number
  /** How many random restarts were needed to get there. */
  restarts: number
  /** The gate scored with the trained network. */
  evaluation: XorEvaluation
}

/**
 * Builds a fresh network for the XOR gate: 2 inputs, a hidden layer and 2
 * outputs (one per gate value, one-hot encoded).
 *
 * @param η The learning rate. Defaults to {@link XOR_LEARNING_RATE}.
 * @returns A network ready to be trained on {@link XOR_DATA}.
 */
export function createXorNetwork(η: number = XOR_LEARNING_RATE): Network {
  return new Network(XOR_LAYER_CONFIG, η, {
    inputSize: XOR_INPUT_SIZE,
    outputSize: XOR_OUTPUT_SIZE,
    batchSize: XOR_BATCH_SIZE,
  })
}

/**
 * Scores a network over the whole gate, running a forward pass per row.
 *
 * @param network The network to score.
 * @param data The dataset to score it on. Defaults to {@link XOR_DATA}.
 * @returns The hits, the misses and one row per sample.
 */
export function evaluateXor(
  network: Network,
  data: TrainingData = XOR_DATA,
): XorEvaluation {
  let hits = 0
  let misses = 0

  const rows: XorRow[] = data.inputs.map((inputs, i) => {
    const expected = data.labels[i]

    network.loadSample(inputs, expected)
    network.feedForward()

    const predicted = network.predictedClass()

    if (predicted === expected) hits++
    else misses++

    return {
      inputs: [...inputs],
      expected,
      predicted,
      outputs: [...network.outputLayer.a],
    }
  })

  return {
    hits,
    misses,
    accuracy: data.inputs.length === 0 ? 0 : hits / data.inputs.length,
    rows,
  }
}

/**
 * Runs one epoch of full-batch gradient descent: every sample's gradient is
 * collected, averaged and applied as a single weight update.
 *
 * @param network The network to train.
 * @param data The dataset to train it on.
 * @returns The mean cost over the dataset, measured after the update.
 */
function trainEpoch(network: Network, data: TrainingData): number {
  const gradients = data.inputs.map((inputs, i) => {
    network.loadSample(inputs, data.labels[i])
    network.feedForward()

    return network.calculateGradient()
  })

  network.backPropagate(network.calculateAverageGradient(gradients))

  // The forward pass above ran with the weights of before the update, so the
  // cost is measured again afterwards to report the epoch's real progress.
  let cost = 0

  timesForEach(data.inputs.length, i => {
    network.loadSample(data.inputs[i], data.labels[i])
    network.feedForward()

    cost += network.cost
  })

  return data.inputs.length === 0 ? 0 : cost / data.inputs.length
}

/**
 * Trains a network on the XOR gate.
 *
 * The gate has four rows, so the whole dataset is one batch and every epoch is
 * a single weight update. XOR's loss surface has local minima that a unlucky
 * initialization can land in, so the training is repeated with fresh random
 * weights and the best run is kept.
 *
 * @param data The dataset to train on. Defaults to {@link XOR_DATA}.
 * @param η The learning rate. Defaults to {@link XOR_LEARNING_RATE}.
 * @param epochs How many epochs each run takes. Defaults to {@link XOR_EPOCHS}.
 * @param restarts How many attempts are made. Defaults to {@link XOR_RESTARTS}.
 * @returns The best run: its network, its cost and how the gate was scored.
 */
export function trainXorNetwork(
  data: TrainingData = XOR_DATA,
  η: number = XOR_LEARNING_RATE,
  epochs: number = XOR_EPOCHS,
  restarts: number = XOR_RESTARTS,
): XorTrainingResult {
  let best: XorTrainingResult | undefined

  timesForEach(Math.max(restarts, 1), restart => {
    const network = createXorNetwork(η)

    let cost = 0

    timesForEach(epochs, () => {
      cost = trainEpoch(network, data)
    })

    const evaluation = evaluateXor(network, data)

    const result: XorTrainingResult = {
      network,
      data,
      cost,
      epochs,
      restarts: restart + 1,
      evaluation,
    }

    // A run is better when it gets more rows right; ties go to the lower cost.
    const isBetter =
      best === undefined ||
      evaluation.accuracy > best.evaluation.accuracy ||
      (evaluation.accuracy === best.evaluation.accuracy && cost < best.cost)

    if (isBetter) best = result
  })

  if (best === undefined) throw new Error('No XOR training run was performed')

  return best
}

/**
 * Prints the gate as a truth table: the inputs, the expected output and what
 * the network predicted.
 *
 * @param evaluation The scored gate.
 * @returns Nothing; the table is logged.
 */
export function printTruthTable(evaluation: XorEvaluation): void {
  console.log('--------')
  console.log('XOR truth table:')
  console.log('--------')

  for (const row of evaluation.rows) {
    console.log({
      inputs: row.inputs.join(' '),
      expected: row.expected,
      predicted: row.predicted,
      outputs: row.outputs.map(value => value.toFixed(4)).join(', '),
      correct: row.expected === row.predicted,
    })
  }

  console.log('--------------')
  console.log('Final results:')
  console.log('--------------')

  console.log({
    hits: evaluation.hits,
    misses: evaluation.misses,
    accuracy: `${formatPercentageWithDecimalPlaces(evaluation.accuracy, 2)}%`,
  })
}

/**
 * Entry point: trains a network on the XOR gate and prints the truth table.
 *
 * @returns Nothing; the training and the results are logged.
 */
const main = () => {
  const result = trainXorNetwork()

  console.log({
    parameters: result.network.parameterCount,
    learningRate: result.network.η,
    epochs: result.epochs,
    restarts: result.restarts,
    cost: result.cost,
  })

  printTruthTable(result.evaluation)
}

if (import.meta.main) main()
