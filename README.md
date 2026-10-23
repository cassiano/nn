# Neural Network from Scratch (TypeScript / Deno)

A hand-written feedforward neural network with two runnable demos: classifying
handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
and learning the XOR gate. No machine learning libraries — the network,
activation functions, matrix math and data loading are all implemented
directly, so you can follow the math behind every equation.

Built with **Deno** + **TypeScript** (fully type-checked, zero runtime deps).

## Status

| Piece                                            | Status                                       |
| ------------------------------------------------ | -------------------------------------------- |
| MNIST loading + parsing                          | ✅ Done                                      |
| Layer / Network classes                          | ✅ Done                                      |
| Forward pass (`z = w·a + b`, activations)        | ✅ Done                                      |
| Cost function (MSE)                              | ✅ Done                                      |
| Per-layer gradient (chain rule, single sample)   | ✅ Done                                      |
| Backpropagation (weight/bias update, mini-batch) | ✅ Done (`Network.backPropagate`)            |
| Batch gradient averaging                         | ✅ Done (`Network.calculateAverageGradient`) |
| XOR training + evaluation                        | ✅ Done (`src/xor.ts`)                       |
| Weight/bias persistence (`Network.saveToDisk`)   | ✅ Done                                      |
| Hyperparameter tuning / accuracy target          | 🔨 In progress                              |

Both demos use the same `Network` class:

- **MNIST** (`src/mnist.ts`) — each sample's gradient is computed
  layer-by-layer, averaged over a batch of 60 and applied to the weights, for
  `NMIST_EPOCHS` epochs (30 by default). Progress and the running accuracy are
  printed per batch while training, the test split is scored image by image at
  the end, and the trained weights are saved to `data/mnist_network.json`.
- **XOR** (`src/xor.ts`) — the four-row gate fits in one batch, so every epoch
  is a single full-batch update. Training runs for `XOR_EPOCHS` epochs (50,000
  by default) and retries up to `XOR_RESTARTS` times with fresh random weights
  to escape local minima. The best run is saved to `data/xor_network.json` and
  printed as a truth table.

Hyperparameters live in `src/constants.ts`.

## How to run

```sh
deno check src/mnist.ts src/xor.ts        # type-check all modules
deno run --allow-read --allow-write src/mnist.ts  # train on MNIST, score the test split, save data/mnist_network.json
deno run --allow-read --allow-write src/xor.ts    # train on XOR and print the truth table, save data/xor_network.json
```

`src/mnist.ts` is the MNIST entry point (previously `src/main.ts`, renamed to
sit alongside `src/xor.ts`).

The MNIST files (.zip) are already shipped inside [`data/mnist/`](./data/mnist).
The loader reads them from disk (no network download) and gunzips them in
memory; `--allow-read` grants access to the dataset files. `--allow-write` is
needed because both entry points save their trained weights under `data/`.
Dataset paths are resolved relative to the loader module, so the commands above
work from any working directory.

### Running the tests

```sh
deno test --allow-read src/   # 184 tests across the whole suite
deno lint                     # type-lint the source and the tests
```

All tests live in [`src/tests/`](./src/tests), with one `*_test.ts` file per
core module, plus a shared `test_helpers.ts` (assertion utilities and small
network harnesses). The demo entry points (`mnist.ts` and `xor.ts`) are not
unit-tested directly; they compose the covered modules.

## Project structure

```
.
├── src/
│   ├── mnist.ts          # MNIST entry point: loads data, trains, scores the test split, saves the weights
│   ├── xor.ts            # XOR dataset + training/evaluation helpers, and its entry point
│   ├── network.ts        # Network class — layers, sample loading, forward pass, cost, gradients, backprop
│   ├── layer.ts          # Layer class — weights/bias, z/w/a computation, per-layer gradient
│   ├── activation.ts     # Activation functions + derivatives (sigmoid, relu, tanh, softmax)
│   ├── mnist_loader.ts   # IDX parsing of the MNIST dataset (+ ASCII image renderer)
│   ├── types.ts          # Shared types (NumericVector, NumericMatrix, TrainingData, Gradient, …)
│   ├── utils.ts          # Loop helpers, matrix ops, random, shuffle, assertions
│   ├── constants.ts      # Hyperparameters and the network topology
│   ├── tests/            # Automated test suite (184 tests) + shared test helpers
│   └── doc_img/          # Screenshots/diagrams referenced from the source comments
├── data/
│   ├── mnist/               # The four zipped MNIST data files (the ones the loader reads)
│   ├── mnist-db/            # The same dataset as per-digit PNG files (not read by the loader)
│   ├── mnist_network.json   # Trained MNIST weights/baises, written by `src/mnist.ts` (gitignored)
│   └── xor_network.json     # Trained XOR weights/baises, written by `src/xor.ts` (gitignored)
└── tsconfig.json         # TS configuration (bundler-style, strict)
```

All the source lives under `src/`; the module imports are relative (`./types.ts`
between modules, `../types.ts` from the tests), so the project can be moved or
renamed without touching a single import path.

## Network architectures

The `Network` class is dataset-agnostic: the topology, input/output sizes,
batch size and learning rate are supplied per demo. Both configurations live in
`src/constants.ts`.

### MNIST — `src/mnist.ts`

Defined by `NMIST_LAYER_CONFIG`:

| Layer    | Neurons            | Activation |
| -------- | ------------------ | ---------- |
| Input    | 784 (28×28 pixels) | —          |
| Hidden 1 | 16                 | relu       |
| Hidden 2 | 16                 | relu       |
| Output   | 10 (digits 0–9)    | softmax    |

Trainable parameters: the two hidden layers plus the output layer each hold a
weight matrix plus a bias vector, initialized randomly in (-1, 1). Total:
**12960 weights + 42 biases = 13002 parameters** (see
`Network.parameterCount`).

MNIST hyperparameters:

| Constant                    | Value | Meaning                                  |
| --------------------------- | ----- | ---------------------------------------- |
| `NMIST_LEARNING_RATE` (η)   | 0.002 | Step size, per sample (see step 6 below) |
| `NMIST_BATCH_SIZE`          | 60    | Samples averaged into one weight update  |
| `NMIST_EPOCHS`              | 30    | Passes over the training set             |

### XOR — `src/xor.ts`

Defined by `XOR_LAYER_CONFIG`:

| Layer  | Neurons                 | Activation |
| ------ | ----------------------- | ---------- |
| Input  | 2 (each input is 0 or 1) | —          |
| Hidden | 4                       | sigmoid    |
| Output | 2 (gate value 0 or 1)   | softmax    |

A hidden layer is required because XOR is not linearly separable: no single
straight line separates `00` and `11` from `01` and `10`. The output is
one-hot encoded, so the gate value `1` is learned as class `1` and `0` as class
`0`. Total: **12 hidden parameters + 10 output parameters = 22 parameters**.

XOR hyperparameters:

| Constant             | Value  | Meaning                                          |
| -------------------- | ------ | ------------------------------------------------ |
| `XOR_LEARNING_RATE`  | 0.5    | Step size, per sample                            |
| `XOR_BATCH_SIZE`     | 4      | Samples averaged into one weight update (all rows) |
| `XOR_EPOCHS`         | 50000  | Passes over the training set                     |
| `XOR_RESTARTS`       | 5      | Fresh random restarts; the best run is kept       |

## How a training step flows through

1. **`main`** (`src/mnist.ts`) — every epoch reshuffles the sample order and
   slices it into batches of `NMIST_BATCH_SIZE` (60). For each sample in a batch
   the label 0–9 is converted to a one-hot vector `y` (`Network.loadSample`),
   e.g. `3 → [0,0,0,1,0,0,0,0,0,0]`. `src/xor.ts` follows the same flow, but
   with the four gate rows as a single full batch.
2. **Forward pass** (`Network.feedForward` → `Layer.calculatePostActivationValues`):
   each hidden/output layer 𝓁 computes its pre-activation
   `z(𝓁) = w(𝓁)·a(𝓁-1) + b(𝓁)` and applies its activation
   `a(𝓁) = σ(z(𝓁))`.
3. **Cost** (`Network.cost`) — mean squared error between the softmax output
   `a(𝐋)` and the target `y`. It sums the per-neuron terms with no 1/n
   normalization, which only rescales the gradient.
4. **Per-sample gradient** (`Network.calculateGradient`) — walks the layers from
   𝐋 back to 1, propagating the error signal
   `δ(𝓁) = ∂C/∂a(𝓁) · σ'(z(𝓁))` and collecting `∂C/∂w(𝓁)` and `∂C/∂b(𝓁)`.
5. **Batch mean** (`Network.calculateAverageGradient`) — one mean gradient per
   batch, each sample weighted 1/60.
6. **Weight update** (`Network.backPropagate`) — steps every layer against that
   mean: `w ← w - η·batchSize·∂C/∂w`, `b ← b - η·batchSize·∂C/∂b`. On MNIST
   η = 0.002 and the batch size is 60; on XOR η = 0.5 and the batch covers all
   4 rows. The batch-size factor cancels the averaging from step 5, so η stays
   a per-sample rate.
7. **Evaluation** — each MNIST test image is run through the network on its own,
   logging every 1000th one, and a final summary reports the overall test
   accuracy. XOR is scored over all four rows and printed as a truth table.
   During training each batch reports its cost, a running average cost and the
   running epoch accuracy, always for the weights _before_ that batch's update.
8. **Persistence** — both entry points finish by calling
   `Network.saveToDisk()`, writing the weights and biases as JSON to
   `data/mnist_network.json` and `data/xor_network.json` respectively.

### Notation used in the code

- `𝓁` — layer index (0 = input layer, 1 = first hidden layer, ..., 𝐋 = last / output layer)
- `w`, `b` — weights, biases
- `z` — pre-activation values
- `a` — activations
- `σ` — activation function; `η` — learning rate
- `y` — expected/one-hot output for the current sample
- `C` — cost (MSE)

Unicode identifiers (`𝓁`, `σ`, `η`, `𝐋`) are used deliberately to keep
equations in code close to their written math form.

## Files in detail

All the paths below are relative to [`src/`](./src).

### `src/mnist.ts`

The MNIST entry point (formerly `src/main.ts`). Loads the dataset with
`MnistLoader`, builds the 784 → 16 → 16 → 10 network from
`NMIST_LAYER_CONFIG`, trains it with mini-batch gradient descent, saves
`data/mnist_network.json`, and finally scores the held-out test split. Progress
is logged per batch and every 1000th test image.

### `src/xor.ts`

The XOR demo. Exports the gate as `XOR_DATA`, plus `createXorNetwork`,
`evaluateXor`, `trainXorNetwork` and `printTruthTable`. `trainXorNetwork` runs
full-batch gradient descent for `XOR_EPOCHS` epochs, restarts with fresh random
weights up to `XOR_RESTARTS` times, and keeps the run with the best accuracy
(ties broken by lower cost). When run directly, it saves
`data/xor_network.json` and prints the learned truth table.

### `src/mnist_loader.ts`

Parses the zipped **IDX** files into `TrainingData` (`inputs`, `labels`), split
into `trainingData` (60k images) and `testData` (10k images). Images are
flattened 28×28 → 784 values, normalized to `[0, 1]` by dividing by 255. Also
ships `MnistLoader.imageAsText('trainingData' | 'testData', i)` (backed by the
static `imageToText`), which renders an image as ASCII art
using the ` ░▒▓▉█` gradient — handy for eyeballing loaded samples.

The four `.zip` files live in [`data/mnist/`](./data/mnist), one level up from
the module; their paths are resolved from the module's own location, so the
loader works from any working directory.

#### IDX format (after gunzip)

| Bytes | Meaning                                           |
| ----- | ------------------------------------------------- |
| 0–3   | Magic number (2051 images / 2049 labels)          |
| 4–7   | Number of records                                 |
| 8–11  | Rows (images)                                     |
| 12–15 | Cols (images)                                     |
| 16+   | Pixel/label data (one byte per value, big-endian) |

### `src/utils.ts`

Tiny functional helpers used throughout:

- `timesForEach` / `timesMap` / `timesReduce` / `timesForEachN` / `timesMapN` /
  `reversedForEach` — index-driven loops and array generators (read nicely and
  keep the math explicit; they mirror the shape of the underlying equations)
- `map` — re-map a value between ranges (with optional clamping), e.g. pixel
  brightness → unicode character index
- `random`, `shuffle` (Fisher–Yates) — randomness utilities
- `createMatrix` / `createVector` — matrices and vectors filled with a repeated
  value or a per-cell function, e.g. the random weight initialization
- `addMatrices` / `addVectors` / `hadamardProduct` — element-wise matrix/vector
  arithmetic (validated against shape mismatches with clear errors)
- `multiplyMatrices` / `multiplyMatrixByVector` / `transposeMatrix` — matrix
  multiplication, the matrix–vector product `w · a` and the transpose
- `dotProduct` / `outerProduct` — the scalar `Σ aᵢbᵢ` and the matrix
  `left[i] · right[j]`, used to build the weight gradient `δ · aᵀ`
- `multiplyMatrixByScalar` / `multiplyVectorByScalar` and their
  `divide…ByScalar` counterparts — scaling a gradient, by the learning rate or
  by the batch size (the divide variants throw on a 0 divisor)
- `gradientAsVector` — flattens a whole gradient into one vector, layers ordered
  by 𝓁 (weights row-major, then biases, per layer)
- `formatPercentageWithDecimalPlaces` — renders a ratio as a percentage rounded
  to a fixed precision, for the logs
- `assertIsNotUndefined` / `assertIsNotNull` / `assertIsNotUndefinedOrNull` —
  type-guard assertions that narrow types at runtime

Every helper returns fresh arrays rather than mutating its operands. The one
exception is `shuffle`, which permutes the array it is given in place.

### `src/activation.ts`

Pure activation functions and their derivatives (derivatives take the
pre-activation `z`, as needed for backprop):
sigmoid, ReLU, tanh and softmax, each numerically stable (sigmoid/softmax
avoid overflow/underflow on extreme inputs). `softmax` reports a constant
derivative of 1, so the output layer's δ stays the plain ∂C/∂a.

### `src/layer.ts`

One layer = one weight matrix `w`, one bias vector `b` and its neuron state
`z`, `a`, `δ`. The input layer only stores raw inputs and ignores
`w`/`b`/`z`/`δ`/`σ`. `calculatePostActivationValues()` recomputes `z` then `a`
for the current sample; the partial derivatives `∂C/∂w`, `∂C/∂b` are built by
`Network.calculateGradient` and stored on the network, not here.

### `src/network.ts`

Ties layers together; owns the current sample's one-hot target `y`, exposes
`inputLayer` / `outputLayer`, drives the forward pass, and computes the cost.
`calculateGradient()` returns one `GradientLayer` per trainable layer;
`calculateAverageGradient()` reduces a batch of them to a single mean gradient;
`backPropagate()` applies that gradient, scaled by η and the network's
`batchSize`, to every layer. `saveToDisk()` persists the trainable layers as
JSON. Note that `calculateAverageGradient` pairs a batch up **by entry
position**, so every sample in a batch must list its layers in the same order —
a sample that disagrees about a layer's `𝓁` at some position is rejected rather
than averaged.

### `src/doc_img/`

Diagrams and screenshots referenced from the source comments as
`[/doc_img/<module>/<timestamp>.png]`. They sit next to the code they
illustrate, so the paths resolve relative to the source file.

## License

Not specified.
