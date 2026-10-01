# Neural Network from Scratch (TypeScript / Deno)

A hand-written feedforward neural network for classifying handwritten digits
from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). No machine
learning libraries — the network, activation functions, matrix math and data
loading are all implemented directly, so you can follow the math behind every
equation.

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
| Hyperparameter tuning / accuracy target          | 🔨 Not started                               |

The network now trains: each sample's gradient is computed layer-by-layer,
averaged over a batch of 60 and applied to the weights, for 5 epochs of
mini-batch gradient descent (see `constants.ts`). Progress and the running
accuracy are printed per batch while training, and the test split is scored
image by image at the end of the run.

## How to run

```sh
deno check main.ts        # type-check all modules
deno run --allow-read main.ts   # load the local MNIST files, train for EPOCHS epochs, score the test split
```

The MNIST files (.zip) are already shipped inside [`data/mnist/`](./data/mnist).
The loader reads them from disk (no network download) and gunzips them in
memory; `--allow-read` grants access to the `./data` directory.

### Running the tests

```sh
deno test --allow-read   # 169 tests across the whole suite
deno lint               # type-lint the source and the tests
```

All tests live in [`tests/`](./tests), one `*_test.ts` file mirroring each
source module, plus a shared `test_helpers.ts` (assertion utilities and small
network harnesses).

## Project structure

```
.
├── main.ts           # Entry point: loads data, builds the network, trains, scores the test split
├── network.ts        # Network class — layers, sample loading, forward pass, cost, gradients, backprop
├── layer.ts          # Layer class — weights/bias, z/w/a computation, per-layer gradient
├── activation.ts     # Activation functions + derivatives (sigmoid, relu, tanh, softmax)
├── mnist_loader.ts   # IDX parsing of the MNIST dataset (+ ASCII image renderer)
├── types.ts          # Shared types (NumericVector, NumericMatrix, TrainingData, Gradient, …)
├── utils.ts          # Loop helpers, matrix ops, random, shuffle, assertions
├── constants.ts      # Hyperparameters and the network topology
├── tests/            # Automated test suite (168 tests) + shared test helpers
├── doc_img/          # Screenshots/diagrams referenced from the source comments
├── tsconfig.json     # TS configuration (bundler-style, strict)
└── data/mnist/       # The four zipped MNIST data files
```

## Network architecture

Defined by `NETWORK_LAYER_CONFIG` in `constants.ts`:

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

Training hyperparameters, all in `constants.ts`:

| Constant                    | Value | Meaning                                  |
| --------------------------- | ----- | ---------------------------------------- |
| `NETWORK_LEARNING_RATE` (η) | 0.002 | Step size, per sample (see step 6 below) |
| `BATCH_SIZE`                | 60    | Samples averaged into one weight update  |
| `EPOCHS`                    | 5     | Passes over the training set             |

## How a training step flows through

1. **`main`**(`main.ts`) — every epoch reshuffles the sample order and slices it
   into batches of `BATCH_SIZE` (60). For each sample in a batch the label 0–9
   is converted to a one-hot vector `y` (`Network.loadSample`), e.g.
   `3 → [0,0,0,1,0,0,0,0,0,0]`.
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
   mean: `w ← w - η·BATCH_SIZE·∂C/∂w`, `b ← b - η·BATCH_SIZE·∂C/∂b`, with
   η = 0.002. The BATCH_SIZE factor cancels the 1/60 from step 5, so η stays a
   per-sample rate.
7. **Evaluation** — each test image is run through the network on its own,
   logging every 1000th one, and a final summary reports the overall test
   accuracy. During training each batch reports its cost, a running average cost
   and the running epoch accuracy, always for the weights _before_ that batch's
   update.

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

### `mnist_loader.ts`

Parses the zipped **IDX** files into `TrainingData` (`inputs`, `labels`), split
into `trainingData` (60k images) and `testData` (10k images). Images are
flattened 28×28 → 784 values, normalized to `[0, 1]` by dividing by 255. Also
ships `MnistLoader.imageAsText('trainingData' | 'testData', i)` (backed by the
static `imageToText`), which renders an image as ASCII art
using the ` ░▒▓▉█` gradient — handy for eyeballing loaded samples.

#### IDX format (after gunzip)

| Bytes | Meaning                                           |
| ----- | ------------------------------------------------- |
| 0–3   | Magic number (2051 images / 2049 labels)          |
| 4–7   | Number of records                                 |
| 8–11  | Rows (images)                                     |
| 12–15 | Cols (images)                                     |
| 16+   | Pixel/label data (one byte per value, big-endian) |

### `utils.ts`

Tiny functional helpers used throughout:

- `timesForEach` / `timesMap` / `timesReduce` / `timesForEachN` / `timesMapN` /
  `reversedForEach` — index-driven loops and array generators (read nicely and
  keep the math explicit; they mirror the shape of the underlying equations)
- `map` — re-map a value between ranges (with optional clamping), e.g. pixel
  brightness → unicode character index
- `random`, `shuffle` (Fisher–Yates) — randomness utilities
- `createMatrix` / `createVector` — matrices and vectors filled with a repeated
  value or a per-cell function, e.g. the random weight initialization
- `toMatrix` / `fromMatrix` — N×1 column-vector conversion
- `addMatrices` / `addVectors` / `hadamardProduct` — element-wise matrix/vector
  arithmetic (validated against shape mismatches with clear errors)
- `multiplyMatrices` / `transposeMatrix` — dot-product and transpose
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

### `activation.ts`

Pure activation functions and their derivatives (derivatives take the
pre-activation `z`, as needed for backprop):
sigmoid, ReLU, tanh and softmax, each numerically stable (sigmoid/softmax
avoid overflow/underflow on extreme inputs). `softmax` reports a constant
derivative of 1, so the output layer's δ stays the plain ∂C/∂a.

### `layer.ts`

One layer = one weight matrix `w`, one bias vector `b` and its neuron state
`z`, `a`, `δ`. The input layer only stores raw inputs and ignores
`w`/`b`/`z`/`δ`/`σ`. `calculatePostActivationValues()` recomputes `z` then `a`
for the current sample; the partial derivatives `∂C/∂w`, `∂C/∂b` are built by
`Network.calculateGradient` and stored on the network, not here.

### `network.ts`

Ties layers together; owns the current sample's one-hot target `y`, exposes
`inputLayer` / `outputLayer`, drives the forward pass, and computes the cost.
`calculateGradient()` returns one `GradientLayer` per trainable layer;
`calculateAverageGradient()` reduces a batch of them to a single mean gradient;
`backPropagate()` applies that gradient, scaled by η and `BATCH_SIZE`, to every
layer. Note that `calculateAverageGradient` pairs a batch up **by entry
position**, so every sample in a batch must list its layers in the same order —
a sample that disagrees about a layer's `𝓁` at some position is rejected rather
than averaged.

## License

Not specified.
