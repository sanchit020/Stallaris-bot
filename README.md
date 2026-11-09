# Stallaris — A Tiny Language Model

**Stallaris** is a character-level, decoder-only Transformer language model implemented in PyTorch and trained entirely from scratch on a custom corpus derived from NASA-related datasets.

The project explores the core principles behind Large Language Models (LLMs), including tokenization, embeddings, causal self-attention, next-token prediction, model training, evaluation, and autoregressive text generation.

No pretrained models, external inference APIs, or third-party language-model checkpoints are used.

## Features

- **Custom Tokenizer:** Character-level tokenizer with vocabulary construction, encoding, and decoding.
- **Dataset Pipeline:** Data loading, cleaning, deduplication, corpus construction, and training/validation splitting.
- **Token and Positional Embeddings:** Represents input tokens and their positions in the sequence.
- **Causal Self-Attention:** Implements masked multi-head self-attention to prevent access to future tokens.
- **Decoder-Only Transformer:** Custom Transformer architecture built using PyTorch.
- **Next-Token Prediction:** Trains the model to predict the next character in a sequence.
- **Training and Evaluation:** Uses cross-entropy loss, AdamW optimization, gradient clipping, and validation loss.
- **Text Generation:** Supports autoregressive generation using temperature sampling and top-k filtering.
- **Interactive Terminal:** Provides a continuous command-line interface for experimenting with generated text.

## Model Architecture

Stallaris uses a compact decoder-only Transformer architecture.

| Component | Configuration |
|---|---:|
| Tokenization | Character-level |
| Vocabulary size | 90 |
| Context length | 128 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| Feed-forward dimension | 512 |
| Dropout | 0.1 |
| Trainable parameters | 436,224 |

### Architecture Flow

```text
Custom Text Corpus
       |
       v
Character Tokenizer
       |
       v
Token + Positional Embeddings
       |
       v
Causal Multi-Head Self-Attention
       |
       v
Feed-Forward Network
       |
       v
Transformer Block (2 Layers)
       |
       v
Final Layer Normalization
       |
       v
Language Model Head
       |
       v
Next-Token Prediction
       |
       v
Autoregressive Text Generation
```

## Dataset

The training corpus was constructed from NASA-related and space-science datasets, including:

- Space-related question-and-answer data
- Exoplanet information
- Extravehicular Activity (EVA) records
- Near-Earth Object (NEO) information

The data preparation pipeline includes cleaning, deduplication, text normalization, document selection, tokenization, and training/validation splitting.

A balanced subset of the available records was selected for model training.

## Training Results

The model was trained for five epochs using AdamW optimization.

| Metric | Result |
|---|---:|
| Final Training Loss | 0.6224 |
| Final Validation Loss | 0.5616 |
| Training Perplexity | 1.86 |
| Validation Perplexity | 1.75 |

The training and validation losses decreased throughout the training process.

*Note: Perplexity is measured at the character-token level and should be interpreted in the context of this model's small vocabulary and structured training corpus.*

## Text Generation

Stallaris generates text autoregressively:

1. Takes an initial text prompt.
2. Converts the prompt into token IDs.
3. Passes the sequence through the Transformer.
4. Predicts a probability distribution for the next token.
5. Samples a token using temperature and top-k filtering.
6. Appends the generated token and repeats.

The model demonstrates the mechanics of language-model generation, but its output is experimental and may contain incoherent or factually incorrect text. It is not intended to function as a reliable question-answering system.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Model Architecture:** Decoder-only Transformer
- **Environment:** Kaggle Notebooks / Jupyter Notebook

## Project Objective

The primary objective of Stallaris is to understand and implement the fundamental architecture and training workflow behind Transformer-based language models.

Rather than relying on pretrained models, the project builds the complete language-model pipeline from the ground up, providing practical experience with tokenization, causal attention, neural network architecture, optimization, evaluation, and autoregressive generation.

## Future Improvements

- Implement conversation-context retention.
- Experiment with larger context windows and model configurations.
- Compare greedy decoding, temperature sampling, and top-k sampling.
- Improve dataset composition and sequence preparation.
- Save and reload trained model checkpoints.

---

**Stallaris — Exploring the architecture behind language models, one token at a time.**
