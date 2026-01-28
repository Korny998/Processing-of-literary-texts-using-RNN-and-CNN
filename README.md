# Russian Literature Author Classification (Keras)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/keras-tf.keras-red.svg)](https://keras.io/)

A Keras / TensorFlow project designed to identify authors of Russian classical literature.  
The system classifies texts from famous Russian writers using multiple neural network architectures, including **SimpleRNN, GRU, LSTM, CNN, and hybrid models**, initialized with **pretrained Navec word embeddings**.

---

## 📁 Project Structure

```bash
├── constants.py        # Global configuration and hyperparameters
├── dataset.py          # Dataset download, preprocessing, tokenization, windowing
├── graphs_example.py   # Visualization utilities (training curves, confusion matrix)
├── models.py           # Neural network architectures
├── model_registry.py   # Registry of all model builder functions
├── trainer.py          # Reusable training logic
├── train.py            # Main training script
├── dataset/            # Auto-created dataset folder
└── README.md           # Project documentation
```

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Visualization](#visualization)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Korny998/Processing-of-literary-texts-using-RNN-and-CNN.git
cd Processing-of-literary-texts-using-RNN-and-CNN
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

**To train all available models and visualize their results, run:**

```python
python train.py
```

The script will automatically:
1. Download and extract the Russian literature dataset (if not present).
2. Tokenize and preprocess all texts.
3. Split texts into overlapping sliding windows.
4. Initialize embedding layers with pretrained Navec vectors.
5. Train all registered models sequentially.
6. Display training curves and confusion matrices for each model.

## Model Architecture

**The project compares multiple approaches to text classification.**

*1. SimpleRNN Models*

Baseline recurrent models for sequence modeling.

- Layers:
    - Embedding (pretrained Navec)
    - SpatialDropout1D
    - BatchNormalization
    - SimpleRNN
    - Dense (Softmax)

*2. GRU Models*

Gated Recurrent Units with stronger temporal modeling.

- Layers:
    - Embedding (pretrained Navec)
    - SpatialDropout1D
    - BatchNormalization
    - GRU
    - Dense (Softmax)
    
*3. LSTM Models*

Long Short-Term Memory networks for long-range dependencies.

- Layers:
    - Embedding (pretrained Navec)
    - SpatialDropout1D
    - BatchNormalization
    - LSTM
    - Dense (Softmax)

*4. CNN Model (Conv1D)*

Long Short-Term Memory networks for long-range dependencies.

- Layers:
    - Embedding
    - Conv1D
    - MaxPooling1D
    - Flatten
    - Dense (Softmax)

*5. Hybrid Models (CNN + RNN / BiLSTM + GRU)*

Long Short-Term Memory networks for long-range dependencies.

- Layers:
    - Embedding
    - Bidirectional LSTM
    - GRU
    - Dense layers with regularization
    - Output Softmax layer

## Dataset

* Data Source

The dataset is automatically downloaded from:

```bash
https://storage.yandexcloud.net/academy.ai/russian_literature.zip
```

* Authors (Classes)

The model classifies texts among: Dostoevsky, Tolstoy, Turgenev, Chekhov, Lermontov, Blok, Pushkin, Gogol, Gorky, Herzen, Bryusov, and Nekrasov.

(Exact set depends on data availability and balancing step.)

* Processing Pipeline

1. Uses a vocabulary of the top MAX_WORDS most frequent tokens.
2. Sliding Window: Long texts are split into overlapping windows of fixed length to increase the training set size.
3. Balancing: Authors with insufficient text length are filtered based on the median sequence length.
4. Embedding Initialization: Word embeddings are initialized using pretrained Navec vectors (300 dimensions).

# Visualization

The graphs_example.py module provides evaluation utilities:

* Training History: Accuracy and loss curves for training and validation data.
* Confusion Matrix: Normalized confusion matrix showing per-author classification performance.