# Neural Network from Scratch (TypeScript / Deno)

A hand-written feedforward neural network for classifying handwritten digits
from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). No machine
learning libraries — the network, activation functions, matrix math and data
loading are all implemented directly, so you can follow the math behind every
equation.

Built with **Deno** + **TypeScript** (fully type-checked, zero runtime deps).

## Status

| Piece | Status |
| --- | --- |
| MNIST loading + parsing | ✅ Done |
| Layer / Network classes | ✅ Done |
| Forward pass (`z = w·a + b`, activations) | ✅ Done |
| Cost function (MSE) | ✅ Done |
| Per-layer gradient (chain rule, single sample) | ✅ Done |
| Backpropagation (accumulating gradients across layers) | 🔨 Placeholder (`Network.backPropagate`) |

The network currently runs a single forward pass over every training sample
but does **not** train: the gradient is computed layer-by-layer yet never used
to update weights.

## How to run

```sh
deno check main.ts        # type-check all modules
deno run --allow-read main.ts   # load the local MNIST files + forward pass over all 60k samples
```

The MNIST files (.zip) are already shipped inside [`mnist/`](./mnist). The
loader reads them from disk (no network download); `--allow-read` grants
access to the `./mnist` directory.

### Running the tests

```sh
deno test --allow-read   # 72 tests across the whole suite
deno lint               # type-lint the source and the tests
```

All tests live in [`tests/`](./tests), one `*_test.ts` file mirroring each
source module, plus a shared `test_helpers.ts` (assertion utilities and small
network harnesses).

## Project structure

```
.
├── main.ts           # Entry point: loads data, builds the network, runs a forward pass
├── network.ts        # Network class — layers, sample loading, forward pass, cost, gradient hook
├── layer.ts          # Layer class — weights/bias, z/w/a computation, per-layer gradient
├── activation.ts     # Activation functions + derivatives (sigmoid, relu, tanh, softmax)
├── mnist_loader.ts   # IDX parsing of the MNIST dataset (+ ASCII image renderer)
├── types.ts          # Shared types (NumericVector, NumericMatrix, TrainingData, …)
├── utils.ts          # Loop helpers, matrix ops, random, shuffle, assertions
├── constants.ts      # Central place for shared constants (empty for now)
├── tests/            # Automated test suite (72 tests) + shared test helpers
├── tsconfig.json     # TS configuration (bundler-style, strict)
└── mnist/            # The four gzipped MNIST data files
```

## Network architecture

Defined in `main.ts`:

| Layer | Neurons | Activation |
| --- | --- | --- |
| Input | 784 (28×28 pixels) | — |
| Hidden 1 | 16 | sigmoid |
| Hidden 2 | 16 | relu |
| Output | 10 (digits 0–9) | softmax |

Trainable parameters: the two hidden layers plus the output layer each hold a
weight matrix plus a bias vector, initialized randomly in (-1, 1). Total:
**12960 weights + 42 biases = 13002 parameters** (see
`Network.parameterCount`).

## How a training sample flows through

1. **`main`**(`main.ts`) — for every image, the label `0–9` is converted to a
   one-hot vector `y` (`Network.loadSample`), e.g. `3 → [0,0,0,1,0,0,0,0,0,0]`.
2. **Forward pass** (`Network.feedForward` → `Layer.calculateActivationValues`):
   each hidden/output layer 𝓁 computes its pre-activation
   `z(𝓁) = w(𝓁)·a(𝓁-1) + b(𝓁)` and applies its activation
   `a(𝓁) = σ(z(𝓁))`.
3. **Cost** (`Network.cost`) — mean squared error between the softmax output
   `a(𝐋)` and the target `y`.

With randomly-initialized weights and no training, the average MSE over the
60,000 training samples ends up around **1.53** (a random-output baseline for
10 classes would be ~0.5; the model has not yet learned at all).

### Notation used in the code

- `𝓁` — layer index (1 = first hidden layer, 𝐋 = last / output layer)
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
Parses the gzipped **IDX** files into `TrainingData` (`inputs`, `labels`).
Images are flattened 28×28 → 784 values, normalized to `[0, 1]` by dividing by
255. Also ships `MnistLoader.imageToText`, which renders an image as ASCII art
using the ` ░▒▓▉█` gradient — handy for eyeballing loaded samples.

#### IDX format (after gunzip)
| Bytes | Meaning |
| --- | --- |
| 0–3 | Magic number (2051 images / 2049 labels) |
| 4–7 | Number of records |
| 8–11 | Rows (images) |
| 12–15 | Cols (images) |
| 16+ | Pixel/label data (one byte per value, big-endian) |

### `utils.ts`
Tiny functional helpers used throughout:

- `timesForEach` / `timesMap` / `timesReduce` / `timesForEachN` / `timesMapN` /
  `reversedForEach` — index-driven loops and array generators (read nicely and
  keep the math explicit; they mirror the shape of the underlying equations)
- `map` — re-map a value between ranges (with optional clamping), e.g. pixel
  brightness → unicode character index
- `random`, `shuffle` (Fisher–Yates) — randomness utilities
- `toMatrix` / `fromMatrix` — N×1 column-vector conversion
- `addMatrices` / `multiplyMatrices` — element-wise add and dot-product
  (validated against dimension mismatches with clear errors)
- `assertIsNotUndefined` / `assertIsNotNull` / `assertIsNotUndefinedOrNull` —
  type-guard assertions that narrow types at runtime

### `activation.ts`
Pure activation functions and their derivatives (derivatives take the
pre-activation `z`, as needed for backprop):
sigmoid, ReLU, tanh and softmax, each numerically stable (sigmoid/softmax
avoid overflow/underflow on extreme inputs).

### `layer.ts`
One layer = one weight matrix `w`, one bias vector `b` and its neuron state.
The input layer only stores raw inputs and ignores `w`/`b`/`z`/`σ`.
`calculateGradient()` computes, for the current sample, the partial
derivatives of the cost w.r.t. every weight and bias of the layer
(`∂C/∂w`, `∂C/∂b`) via the chain rule.

### `network.ts`
Ties layers together; owns the current sample's one-hot target `y`, exposes
`inputLayer` / `outputLayer`, drives the forward pass, and computes the cost.
`backPropagate()` is the declared hook for learning (not implemented yet).

## License

Not specified.