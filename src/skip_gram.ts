import { Network } from './network.ts'
import { getWordPairs, createMatrix, shuffle, random } from './utils.ts'
import {
  SKIP_GRAM_WORDS,
  SKIP_GRAM_WINDOW_SIZE,
  SKIP_GRAM_EPOCHS,
} from './constants.ts'
import {
  SKIP_GRAM_LEARNING_RATE,
  SKIP_GRAM_BATCH_SIZE,
  SKIP_GRAM_LAYER_CONFIG,
  SKIP_GRAM_VOCAB_SIZE,
} from './constants.ts'

export function createSkipGramNetwork(
  η: number = SKIP_GRAM_LEARNING_RATE,
  batchSize: number = SKIP_GRAM_BATCH_SIZE,
): Network {
  return new Network(SKIP_GRAM_LAYER_CONFIG, η, {
    inputSize: SKIP_GRAM_VOCAB_SIZE,
    outputSize: SKIP_GRAM_VOCAB_SIZE,
    batchSize,
    useBias: false,
    costFunction: 'cross-entropy',
  })
}

/**
 * Reinitializes the weights of all non-input layers with small random values.
 *
 * @param network The network whose weights should be reinitialized.
 * @param scale The scale for random values, weights will be in [-scale, scale].
 * @returns Nothing; weights are updated in place.
 */
export function reinitializeNetworkWeights(
  network: Network,
  scale: number = 0.01,
): void {
  for (let i = 1; i < network.layers.length; i++) {
    const layer = network.layers[i]

    layer.w = createMatrix(layer.size, layer.previousLayer.size, () =>
      random(-scale, scale),
    )
  }
}

/**
 * Generates skip-gram word pairs from the vocabulary and window size,
 * displays them to the console, and returns them.
 *
 * @returns Array of [inputIndex, contextIndex] pairs from getWordPairs.
 */
export function getSkipGramPairs(): [number, number][] {
  const pairs = getWordPairs(SKIP_GRAM_WORDS, SKIP_GRAM_WINDOW_SIZE)

  return pairs
}

/**
 * Converts a word index to a one-hot encoded vector.
 *
 * @param index The index of the word in the vocabulary.
 * @returns A one-hot vector of length SKIP_GRAM_VOCAB_SIZE.
 */
export function wordToOneHot(index: number): number[] {
  const vector = Array(SKIP_GRAM_VOCAB_SIZE).fill(0)
  vector[index] = 1

  return vector
}

export function trainSkipGramNetwork(
  epochs: number = SKIP_GRAM_EPOCHS,
  batchSize: number = SKIP_GRAM_BATCH_SIZE,
  η: number = SKIP_GRAM_LEARNING_RATE,
  verbose: boolean = true,
): Network {
  const network = createSkipGramNetwork(η, batchSize)
  const pairs = getSkipGramPairs()

  console.log('Word pairs:')
  console.log(
    pairs
      .map(([i, j]) => `(${SKIP_GRAM_WORDS[i]}, ${SKIP_GRAM_WORDS[j]})`)
      .join(' '),
  )

  for (let epoch = 0; epoch < epochs; epoch++) {
    const shuffledPairs = [...pairs]
    shuffle(shuffledPairs)

    for (let i = 0; i < shuffledPairs.length; i += batchSize) {
      const batchPairs = shuffledPairs.slice(i, i + batchSize)
      const gradients = []

      for (const [inputIndex, targetIndex] of batchPairs) {
        const input = wordToOneHot(inputIndex)

        network.loadSample(input, targetIndex)
        network.feedForward()

        gradients.push(network.calculateGradient())
      }

      const avgGradient = network.calculateAverageGradient(gradients)

      network.backPropagate(avgGradient)
    }

    if (
      verbose &&
      (epoch === 0 ||
        (epoch + 1) % Math.max(1, Math.floor(epochs / 10)) === 0 ||
        epoch + 1 === epochs)
    ) {
      let totalCost = 0

      for (const [inputIndex, targetIndex] of pairs) {
        network.loadSample(wordToOneHot(inputIndex), targetIndex)
        network.feedForward()

        totalCost += network.cost
      }

      const avgCost = totalCost / pairs.length

      console.log(
        `Epoch ${epoch + 1}/${epochs} - Average Cost: ${avgCost.toFixed(6)}`,
      )
    }
  }

  return network
}

export function getWordEmbeddings(network: Network): number[][] {
  const hiddenLayer = network.layers[1]
  const embeddings: number[][] = Array(SKIP_GRAM_VOCAB_SIZE)
  const vocabSize = SKIP_GRAM_VOCAB_SIZE
  const embeddingDim = hiddenLayer.size

  for (let col = 0; col < vocabSize; col++) {
    const embedding = Array(embeddingDim)

    for (let row = 0; row < embeddingDim; row++)
      embedding[row] = hiddenLayer.w[row][col]

    embeddings[col] = embedding
  }

  return embeddings
}

/**
 * Computes the cosine similarity between two vectors.
 *
 * @param a First vector.
 * @param b Second vector.
 * @returns Cosine similarity score in [-1, 1].
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length)
    throw new Error('Vectors must have the same length')

  let dotProduct = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]

    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }

  if (normA === 0 || normB === 0) return 0

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}

/**
 * Compares two vectors using cosine similarity.
 *
 * @param a First vector.
 * @param b Second vector.
 * @returns Cosine similarity score in [-1, 1].
 */
export function compareVectors(a: number[], b: number[]): number {
  return cosineSimilarity(a, b)
}

export function getWordEmbeddingsByWord(
  network: Network,
): Record<string, number[]> {
  const embeddings = getWordEmbeddings(network)
  const result: Record<string, number[]> = {}

  SKIP_GRAM_WORDS.forEach((word, index) => {
    result[word] = embeddings[index]
  })

  return result
}

/**
 * Finds the most similar words to a target word using cosine similarity.
 *
 * @param network The trained skip-gram network.
 * @param targetWord The word to find similar words for.
 * @param topK The number of similar words to return.
 * @returns Array of words with their cosine similarity scores.
 */
export function findMostSimilar(
  network: Network,
  targetWord: string,
  topK: number = 5,
): Array<{ word: string; similarity: number }> {
  const embeddingsByWord = getWordEmbeddingsByWord(network)
  const targetEmbedding = embeddingsByWord[targetWord]

  if (!targetEmbedding)
    throw new Error(`Word "${targetWord}" not in vocabulary`)

  const similarities: Array<{ word: string; similarity: number }> = []

  for (const word of Object.keys(embeddingsByWord)) {
    if (word === targetWord) continue

    const similarity = cosineSimilarity(targetEmbedding, embeddingsByWord[word])
    similarities.push({ word, similarity })
  }

  similarities.sort((a, b) => b.similarity - a.similarity)

  return similarities.slice(0, topK)
}

/**
 * Generates and displays a report showing word similarities using cosine similarity.
 *
 * @param network The trained skip-gram network.
 * @returns Nothing; outputs the report to console.
 */
export function generateSimilarityReport(network: Network): void {
  console.log('\n=== Word Similarity Report (Cosine Similarity) ===\n')

  for (const word of SKIP_GRAM_WORDS) {
    const similar = findMostSimilar(network, word, 5)

    console.log(`${word}:`)

    for (const { word: similarWord, similarity } of similar)
      console.log(`  ${similarWord.padEnd(10)} ${similarity.toFixed(4)}`)

    console.log()
  }
}

const main = async () => {
  console.log('Training Skip-Gram model...')
  console.log(`Vocabulary: ${SKIP_GRAM_WORDS.join(', ')}`)
  console.log(`Window size: ${SKIP_GRAM_WINDOW_SIZE}`)

  const pairs = getSkipGramPairs()

  console.log(`Number of word pairs: ${pairs.length}`)

  const network = trainSkipGramNetwork(
    SKIP_GRAM_EPOCHS,
    SKIP_GRAM_BATCH_SIZE,
    SKIP_GRAM_LEARNING_RATE,
  )

  await network.saveToDisk('skip_gram_network.json')

  console.log('\nTraining complete!')
  console.log(`Word embeddings (hidden layer weight matrix):`)

  const embeddings = getWordEmbeddings(network)

  SKIP_GRAM_WORDS.forEach((word, index) => {
    console.log(
      `  ${word} (${index}): [${embeddings[index]
        .slice(0, 5)
        .map(x => x.toFixed(4))
        .join(', ')}...] (dim=${embeddings[index].length})`,
    )
  })

  generateSimilarityReport(network)
}

if (import.meta.main) await main()
