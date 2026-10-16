
# Echoes Beyond Earth: Sci-Fi Text Generation Using RNN Architectures

## 1. Problem Statement and Objective

Autoregressive text generation predicts the next word based on previously observed words and generates text sequentially. This project implements and compares four recurrent neural network architectures for science-fiction text generation: Simple RNN, Bidirectional RNN, LSTM, and GRU.

The objectives are to preprocess a textual dataset, construct input-target sequences, train and evaluate all four models, compare their performance, and analyse the influence of sampling temperature on generated text.

## 2. Dataset Description and Source

The project uses three science-fiction texts:

- The Radio Planet
- The Blue Star
- Second Variety

The combined corpus contains approximately **175,872 words** after whitespace normalization. It includes narrative prose, dialogue, science-fiction terminology, and descriptions of characters and settings.

**Dataset source:** [https://www.kaggle.com/datasets/atharvaatulrege/quantumquill-scifi-dataset](https://www.kaggle.com/datasets/atharvaatulrege/quantumquill-scifi-dataset)

### Preprocessing

1. Loaded and combined the three selected text files.
2. Normalized whitespace and removed the leading title/byline from *Second Variety*.
3. Tokenized the text using the Keras Tokenizer.
4. Limited the vocabulary to 8,000 tokens, including the out-of-vocabulary token.
5. Used a sequence length of 40 tokens.
6. Split the token stream into training, validation, and test sets using an 80:10:10 ratio.
7. Constructed input sequences of 40 tokens and corresponding next-word targets.

| Dataset Split | Number of Sequences |
|---|---:|
| Training | 143,044 |
| Validation | 17,886 |
| Testing | 17,886 |

## 3. Models Implemented

All models use an embedding layer, the Adam optimizer, and sparse categorical cross-entropy loss. Each model predicts a probability distribution over the next word.

### Simple RNN
Uses a recurrent layer to learn sequential patterns from previous tokens. It serves as the baseline model.

### Bidirectional RNN
Processes sequences in forward and backward directions during training to learn representations from both directions.

### LSTM
Uses input, forget, and output gates to retain relevant information and learn dependencies across sequences.

### GRU
Uses update and reset gates to control information flow with a simpler gated architecture than LSTM.

### Common Configuration

- Vocabulary size: 8,000
- Sequence length: 40
- Embedding dimension: 64
- Recurrent units: 64 for Simple RNN, LSTM, and GRU
- Bidirectional RNN: 32 units per direction
- Optimizer: Adam
- Loss function: Sparse categorical cross-entropy
- Batch size: 128
- Training: Initial training followed by continued training with early stopping

## 4. Dependencies

- Python 3
- TensorFlow / Keras
- NumPy
- pandas
- Matplotlib
- Seaborn

Install the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

## 5. Execution Steps

1. Open the notebook in Kaggle or another compatible Python environment.
2. Attach the dataset containing the selected text files.
3. Execute the cells in order to load and preprocess the text.
4. Construct the input-target sequences.
5. Build and train the four models.
6. Evaluate the models on the test set.
7. Generate text using the same seed at different temperatures.
8. Execute the plotting and export cells to save the results.

## 6. Quantitative Results

The following results were obtained from the trained models.

| Model | Test Loss | Accuracy (%) | Perplexity | Training Time (s) |
|---|---:|---:|---:|---:|
| Simple RNN | 6.7376 | 10.29 | 843.50 | 408.02 |
| Bidirectional RNN | 6.7372 | 9.62 | 843.22 | 400.94 |
| LSTM | 6.9438 | 9.80 | 1036.68 | 853.86 |
| GRU | 6.7711 | 10.09 | 872.31 | 879.94 |

### Evaluation Metrics

- **Test Loss:** Measures the error in next-word prediction.
- **Accuracy:** Measures the percentage of correctly predicted next words.
- **Perplexity:** Measures uncertainty in next-word predictions; lower values are preferable.
- **Training Time:** Measures the time required to train the models.

## 7. Model Comparison

The Bidirectional RNN achieved the lowest test loss and perplexity. The Simple RNN achieved the highest prediction accuracy and required less training time than the LSTM and GRU.

Qualitative evaluation considered:

- **Coherence:** Whether generated sentences form meaningful passages.
- **Diversity:** Variation in generated words and phrases.
- **Contextual Consistency:** Whether the text maintains its topic and narrative context.
- **Repetition:** Whether words and phrases are unnecessarily repeated.

The generated samples contained some locally plausible phrases, but narrative coherence and contextual consistency varied. Repeated phrases and out-of-vocabulary markers also appeared in several outputs.

## 8. Sampling-Temperature Analysis

The same seed sequence was used for all models:

`the Russian soldier made his way`

Text was generated at temperatures **0.3, 0.7, and 1.0**.

### Temperature 0.3
Lower temperature favours higher-probability words, producing more predictable output. However, repeated phrases and `<UNK>` tokens appeared frequently.

### Temperature 0.7
The generated text showed greater variation in vocabulary and phrasing, although grammatical and contextual consistency remained limited.

### Temperature 1.0
Higher temperature produced more unpredictable word choices and greater variation, but the generated passages were less consistent.

Overall, increasing temperature encouraged diversity but did not necessarily improve coherence. The appropriate temperature depends on the desired balance between predictability and variation.

## 9. Conclusion

This project implemented and compared Simple RNN, Bidirectional RNN, LSTM, and GRU models for autoregressive science-fiction text generation.

The **Bidirectional RNN** is selected as the best-performing model according to test loss and perplexity. The Simple RNN is an alternative when prediction accuracy and computational efficiency are priorities.

The temperature experiment demonstrates that sampling temperature affects text diversity, predictability, and repetition. The results also illustrate that next-word prediction metrics and narrative coherence measure different aspects of text-generation performance.

## 10. Output Files

The `assignment_results` directory contains:

- `model_comparison.csv` — quantitative model comparison
- `generated_samples.csv` — generated text at three temperatures
- `test_loss.png` — test-loss comparison
- `accuracy.png` — test-accuracy comparison
- `perplexity.png` — perplexity comparison
