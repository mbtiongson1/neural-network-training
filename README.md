# Implementing the Backpropagation Algorithm

A Multilayer Perceptron (MLP) Neural Network trained using the Backpropagation Algorithm on a challenging 8-class dataset. This project explores the full pipeline — from data preprocessing and class balancing, through architecture design and mathematical formulation, to iterative hyperparameter tuning and model evaluation.

> **Course:** AI 201 — 2S AY 2025–2026
> **Assignment:** Programming Assignment 3 — Neural Network

---

## Project Structure

```
PA3 - Neural Network/
├── LICENSE
├── README.md
├── .gitignore
├── src/                          # Additional source files
├── python/                       # Modular Python refactor of main.ipynb
│   ├── main.py                   # Main execution engine (run this)
│   ├── activations.py            # Activation functions and derivatives
│   ├── network.py                # OutputLayer, HiddenLayer, Epoch classes
│   ├── utils.py                  # Utilities: Partition, train, predictions, charts
│   ├── checkscores.py            # Score aggregation tool
│   ├── dataset/                  # Copy of the dataset for standalone execution
│   ├── figures/                  # Generated charts and learning curves
│   ├── export/                   # Training artifacts (generated at runtime)
│   ├── modelA/                   # Best model weights (generated at runtime)
│   ├── modelB/                   # Backup model weights (generated at runtime)
│   └── predictions/              # Test set predictions (generated at runtime)
└── submission/                   # Original notebook and raw training outputs
    ├── main.ipynb                # Original Jupyter notebook
    ├── checkscores.py            # Original score aggregation script
    ├── dataset/                  # Original dataset CSVs
    ├── export/                   # Original training exports
    ├── final/                    # Final combined scores
    ├── modelA/                   # Original best model weights
    ├── modelB/                   # Original backup model weights
    ├── predictions/              # Original test set predictions
    └── combined_scores.csv       # Aggregated scores CSV
```

### Running the Python Version

```bash
cd python/
python main.py           # Train all networks and export predictions
python checkscores.py    # Aggregate and rank scores from export/
```

---

## Table of Contents

- [Project Structure](#project-structure)
- [Initializing the Libraries](#initializing-the-libraries)
- [The Dataset](#the-dataset)
  - [Dataset Distribution](#dataset-distribution)
  - [SMOTE (Synthetic Minority Over-sampling Technique)](#smote-synthetic-minority-over-sampling-technique)
  - [Partitioning the Dataset](#partitioning-the-dataset)
  - [Batch Size](#batch-size)
- [Setting up the Equations](#setting-up-the-equations)
  - [Forward Propagation](#1-forward-propagation)
  - [Activation Functions](#2-choosing-the-activation-functions)
  - [Error & Mean Square Error](#3-error--mean-square-error)
  - [Backpropagation Delta Equations](#4-backpropagation-delta-equations)
  - [Weight Update with Momentum](#5-weight-update-negative-gradient-with-momentum)
- [Training Phase](#training-phase)
  - [Hyperparameters Setup](#hyperparameters-setup)
  - [Training: Network A](#training-network-a)
  - [Training: Network B](#training-network-b)
- [Training Results](#training-results)
- [Loading the Model & Running Predictions](#loading-the-model)
- [Conclusion and Recommendations](#conclusion-and-recommendations)

---

## Initializing the Libraries

The following libraries are required:

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical operations, matrix math, vectorized computations |
| `matplotlib` | Plotting learning curves and dataset distributions |
| `imblearn` | SMOTE for handling class imbalance (depends on `scipy`) |
| `os`, `csv` | File I/O for loading datasets and exporting results |
| `time` | Tracking training duration |

You can install these dependencies using the provided `requirements.txt` file within the `python` directory:

```bash
cd python/
pip install -r requirements.txt
```

---

## The Dataset

The dataset is loaded from CSV files:

```python
DATASET    = np.loadtxt("dataset/data.csv", delimiter=",")       # (N, 2052) features
DATALABELS = np.loadtxt("dataset/data_labels.csv", dtype=int)    # (N,) classes 1–8
TESTSET    = np.loadtxt("dataset/test_set.csv", delimiter=",")   # unlabeled test set
```

Labels are one-hot encoded into an `(N, 8)` matrix for the output layer — each row has a `1.0` at the index corresponding to its class.

### Dataset Distribution

An initial analysis of the class distribution revealed a significant imbalance. A pie chart visualization showed that **Class 1 dominates with 46.6%** of the dataset, while **Class 4 holds the smallest share**. This kind of skew can cause the network to become biased toward the majority class, predicting it disproportionately and ignoring minority classes entirely.

### SMOTE (Synthetic Minority Over-sampling Technique)

To address the class imbalance, **SMOTE** was applied to the training data:

```python
from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE(random_state=50).fit_resample(X, y)
```

After SMOTE, the dataset is perfectly balanced — each class has an equal number of samples. A follow-up pie chart confirmed the uniform distribution.

> **Note on data leakage:** In this implementation, SMOTE was applied *before* splitting into training and validation sets. This means synthetic samples could influence the validation set, potentially leading to optimistic evaluation metrics. Ideally, SMOTE should only be applied to the training fold after the split. This trade-off was accepted for simplicity but is worth noting.

### Partitioning the Dataset

A custom `Partition` class handles the train/validation split:

- **Validation set:** 800 samples (randomly selected, `random_state=50`)
- **Training set:** remaining samples
- Partitions are exported to CSV files (`training_set.csv`, `training_labels.csv`, `validation_set.csv`, `validation_labels.csv`) for reproducibility.

```python
split = Partition(X_balanced, y_balanced)
```

### Batch Size

Mini-batch gradient descent is used with a **batch size of 8**. A generator function `minibatch()` yields sequential slices of the training set and labels for each batch iteration.

---

## Setting up the Equations

The following notation is used throughout:

| Symbol | Meaning |
|--------|---------|
| $L$ | Total number of layers (output layer index) |
| $l$ | Current layer index, $l \in \{1, \dots, L\}$ |
| $i$ | Index of neuron in layer $l-1$; $i = 0$ is the bias input |
| $j$ | Index of neuron in layer $l$ |
| $k$ | Index of neuron in output layer $L$ |
| $w^{(l)}_{ji}$ | Weight from neuron $i$ in layer $l-1$ to neuron $j$ in layer $l$ |
| $v^{(l)}_j$ | Internal activity (pre-activation) of neuron $j$ in layer $l$ |
| $\varphi_j$ | Activation function of neuron $j$ |
| $d_k$ | Desired (target) output for output neuron $k$ |
| $e_k(n)$ | Error at output neuron $k$ at epoch $n$ |
| $\mathcal{E}(n)$ | Mean square error over all output neurons at epoch $n$ |
| $\eta$ | Learning rate |
| $\alpha$ | Momentum coefficient |
| $\gamma$ | Leaky ReLU negative-side slope |

### 1. Forward Propagation

#### 1.1 Internal Activity

The weighted sum includes the bias weight $w^{(l)}_{j0}$ (with fixed input $\varphi_0 = +1$):

$$v^{(l)}_j(n) = \sum_{i=0}^{p} w^{(l)}_{ji}(n)\, \varphi_i\!\left(v^{(l-1)}_i(n)\right)$$

#### 1.2 Neuron Output

$$\varphi_j\!\left(v^{(l)}_j(n)\right)$$

where $\varphi_j$ is the chosen activation function.

### 2. Choosing the Activation Functions

Derivatives are expressed in terms of the neuron output $o(n) = \varphi(v(n))$ directly.

#### 2.1 Logistic (Sigmoid)

$$\varphi(v) = \frac{1}{1 + e^{-av}}, \quad a > 0$$

**Derivative:**

$$\varphi'(v) = a \cdot o(n)\,\bigl(1 - o(n)\bigr)$$

> Given: $a = 2.0$

#### 2.2 Hyperbolic Tangent (Tanh)

$$\varphi(v) = a \tanh(bv)$$

**Derivative:**

$$\varphi'(v) = \frac{b}{a}\left(a - o(n)\right)\!\left(a + o(n)\right)$$

> Given: $a = 1.716,\; b = 2/3$ (LeCun's recommended parameters)

#### 2.3 Leaky ReLU

$$\varphi(v) = \begin{cases} v & \text{if } v > 0 \\ \gamma\, v & \text{if } v \leq 0 \end{cases}, \quad \gamma \in (0,1)$$

**Derivative:**

$$\varphi'(v) = \begin{cases} 1 & \text{if } o(n) > 0 \\ \gamma & \text{if } o(n) \leq 0 \end{cases}$$

### 3. Error & Mean Square Error

#### 3.1 Error Signal at Output Neuron $k$

$$e_k(n) = d_k(n) - \varphi_k\!\left(v^{(L)}_k(n)\right)$$

#### 3.2 Mean Square Error

$$\mathcal{E}(n) = \frac{1}{2} \sum_{k \in \mathcal{C}} e_k^2(n)$$

### 4. Backpropagation Delta Equations

#### 4.1 Output Layer Delta

$$\delta^{(L)}_k(n) = e_k(n) \cdot \varphi'\!\left(v^{(L)}_k(n)\right)$$

#### 4.2 Hidden Layer Delta

For layer $l \in \{L-1,\, L-2,\, \dots,\, 1\}$:

$$\delta^{(l)}_j(n) = \varphi'\!\left(v^{(l)}_j(n)\right) \cdot \sum_{k}\, \delta^{(l+1)}_k(n)\, w^{(l+1)}_{kj}(n)$$

The hidden-layer delta propagates the error signal backward by weighting each upstream delta by the corresponding weight and scaling by the local derivative.

### 5. Weight Update: Negative Gradient with Momentum

The weight correction follows the **negative gradient** of $\mathcal{E}(n)$:

$$-\frac{\partial \mathcal{E}(n)}{\partial w^{(l)}_{ji}(n)} = \delta^{(l)}_j(n) \cdot \varphi_i\!\left(v^{(l-1)}_i(n)\right)$$

#### 5.1 Weight Increment (Generalized Delta Rule + Momentum)

$$\Delta w^{(l)}_{ji}(n) = \alpha\, \Delta w^{(l)}_{ji}(n-1) + \eta\, \delta^{(l)}_j(n)\, \varphi_i\!\left(v^{(l-1)}_i(n)\right)$$

#### 5.2 Weight Update Rule

$$w^{(l)}_{ji}(n+1) = w^{(l)}_{ji}(n) + \Delta w^{(l)}_{ji}(n)$$

> **Bias weights** ($i = 0$) are updated identically using $\varphi_0 = +1$, so no separate update rule is needed.

> **Momentum note:** When $\alpha = 0$, this reduces to the standard (vanilla) delta rule. When $\alpha > 0$, the update accumulates a fraction of the previous step, smoothing oscillations and accelerating convergence along consistent gradient directions.

---

## Training Phase

### Hyperparameters Setup

Two network configurations were defined, differing primarily in their activation functions:

| Parameter | Network A (Tanh) | Network B (Leaky ReLU) |
|-----------|-------------------|------------------------|
| Hidden activation | Tanh ($a=1.716, b=2/3$) | Leaky ReLU ($\gamma=0.01$) |
| Output activation | Logistic ($a=1.0$) | Logistic ($a=1.0$) |
| Learning rate ($\eta$) | 0.85 | 0.85 |
| Momentum ($\alpha$) | 0.9 | 0.9 |
| Hidden layer size | 8 | 8 |
| Batch size | 8 | 8 |

The architecture uses **two hidden layers** (layers $i$ and $j$) plus an **output layer** ($k$), with a fixed structure of input → hidden → hidden → output (8 classes).

Training runs for **100 epochs** by default, printing progress every 5 epochs. After training, the `Scores` method computes per-class metrics (TP, TN, FP, FN, Precision, Recall, F1) and summary metrics (Accuracy, Macro-Averaged F1, Matthews Correlation Coefficient).

### Training: Network A

#### Default Parameters — First Attempt

The initial training with the default hyperparameters (`eta=0.85`, `alpha=0.9`, `size=8`) did not converge. The error remained essentially flat across all 100 epochs, and the resulting F1 score was a dismal **0.0324** — the network was effectively guessing a single class.

This indicated that the learning rate was too aggressive and the momentum too high, causing the weight updates to overshoot and oscillate without settling into a useful region of the loss surface.

A second run with the same parameters confirmed the issue — the F1 score dropped even further to **0.0283**. Clearly, the defaults were not viable.

#### Improvements for Network A

The hyperparameters were adjusted based on the observation that convergence was failing:

| Parameter | Default | Improved |
|-----------|---------|----------|
| $\eta$ | 0.85 | **0.1** |
| $\alpha$ | 0.9 | **0.5** |
| Hidden size | 8 | **12** |

```python
NetworkA['eta'] = 0.1    # lowering the learning rate
NetworkA['alpha'] = 0.5  # lowering the momentum
NetworkA['size'] = 12    # increasing hidden layer nodes
```

The results improved dramatically — errors converged steadily, and the validation error aligned well with the training error. The F1 score jumped to **0.9744**. Training time roughly doubled (from ~38s to ~43s) due to the larger hidden layer, but since convergence happened early, training could reasonably be stopped around the **40th epoch**.

#### Further Tuning — Faster Network A

To test whether a slightly smaller hidden layer could deliver comparable results faster, the size was reduced back to 10:

```python
NetworkA['size'] = 10  # trying a smaller hidden layer
```

Surprisingly, this configuration achieved a **lower MSE of 0.00083** and a marginally better F1 score of **0.9748**, while being as fast as the default configuration (~40.8s). Having only 10 hidden nodes proved to be a sweet spot — large enough to capture the decision boundaries, small enough to train efficiently.

### Training: Network B

#### Default Parameters — First Attempt

Network B, using Leaky ReLU for hidden layers, was trained with the same default parameters (`eta=0.85`, `alpha=0.9`, `size=8`). The result was identical to Network A's failure — the errors did not change, and the F1 score was only **0.0263**. The default learning rate was simply too high for this architecture as well.

#### Improvements for Network B

The same tuning strategy that worked for Network A was applied:

```python
NetworkB['eta'] = 0.1    # decreasing the learning rate
NetworkB['alpha'] = 0.5  # decreasing the momentum
NetworkB['size'] = 10    # increasing hidden layer nodes
```

This produced the **best result of all configurations** — an F1 score of **0.9866** with 98.6% accuracy. The learning curve showed excellent convergence with training and validation errors tracking closely. The small time penalty (~43.6s vs ~40.8s for Network A fast) was considered well worth the performance gain.

---

## Training Results

### F1 Score Rankings

| Rank | Network | F1 Score | Time (s) |
|------|---------|----------|----------|
| 🥇 1 | Network B (improved) — Leaky ReLU | **0.9866** | 43.57 |
| 🥈 2 | Network A (improved, fast) — Tanh | 0.9748 | 40.83 |
| 🥉 3 | Network A (improved) — Tanh | 0.9744 | 42.88 |
| 4 | Network A (default, run 1) | 0.0324 | 37.72 |
| 5 | Network A (default, run 2) | 0.0283 | 38.63 |
| 6 | Network B (default) | 0.0263 | 41.89 |

### Full Scores

| Rank | Network | Accuracy | Precision | Recall | F1 | MCC |
|------|---------|----------|-----------|--------|------|------|
| 1 | networkB_improv | 0.98625 | 0.98646 | 0.98683 | 0.98660 | 0.98428 |
| 2 | networkA_improv_fast | 0.97500 | 0.97591 | 0.97548 | 0.97476 | 0.97167 |
| 3 | networkA_improv | 0.97375 | 0.97602 | 0.97472 | 0.97439 | 0.97026 |
| 4 | networkA (run 1) | 0.14875 | 0.01859 | 0.12500 | 0.03237 | 0.00000 |
| 5 | networkA (run 2) | 0.12750 | 0.01594 | 0.12500 | 0.02827 | 0.00000 |
| 6 | networkB | 0.11750 | 0.01469 | 0.12500 | 0.02629 | 0.00000 |

**Network B (improved)** with Leaky ReLU is the clear winner at **98.6% accuracy and 0.9866 F1 score**. Its trained weights were exported to `modelA/trained_weights.csv` as the primary model for unseen test data.

**Network A (improved, fast)** with Tanh came in second at **97.5% accuracy and 0.9748 F1**. It loads faster and would scale better on longer epoch iterations. Its weights were exported to `modelB/trained_weights.csv` as the backup model.

The bottom three configurations (all using default hyperparameters) achieved near-zero F1 scores with MCC of 0.0, confirming they learned nothing meaningful — they simply predicted the majority class.

---

## Loading the Model

Trained weights are loaded from CSV files using the `loadWeights` function, which reconstructs the weight matrices for each layer.

## Running and Exporting Predictions

The `runPredictions` function takes the loaded weights and the test set, reconstructs the network layers, and performs a forward pass for each test sample. Predictions are determined by taking the `argmax` of the output layer activations.

The best two models were used to generate predictions:

- **Network B improved** (Leaky ReLU) → `predictions/networkB_improv_predictions.csv`
- **Network A improved fast** (Tanh) → `predictions/networkA_improv_fast_predictions.csv`

---

## Conclusion and Recommendations

This project demonstrated the critical importance of **hyperparameter tuning** in neural network training. The default configuration (`eta=0.85`, `alpha=0.9`) failed completely across both network architectures, while a simple reduction to `eta=0.1` and `alpha=0.5` transformed the results from near-random guessing (~3% F1) to high-accuracy classification (~98% F1).

**Key takeaways:**

1. **Learning rate dominates early performance.** The default rate of 0.85 was far too aggressive, causing the networks to overshoot and fail to converge. Reducing it to 0.1 was the single most impactful change.

2. **Leaky ReLU outperformed Tanh.** Network B (improved) with Leaky ReLU achieved 0.9866 F1 versus Network A's best of 0.9748. The Leaky ReLU's ability to maintain gradient flow for negative inputs likely contributed to better training dynamics.

3. **Hidden layer size has diminishing returns.** Increasing from 8 to 12 nodes helped convergence, but reducing to 10 nodes actually produced slightly better results with faster training. Over-parameterization was not necessary for this dataset.

4. **Early stopping is recommended.** The learning curves showed convergence around epoch 40 for the improved configurations. Training the full 100 epochs is unnecessary and risks overfitting.

5. **SMOTE effectively addressed class imbalance.** Without balancing, the network would likely have remained biased toward the majority class (Class 1 at 46.6%). The balanced dataset enabled the network to learn meaningful boundaries for all 8 classes. However, applying SMOTE before the train/validation split introduced potential data leakage — a future improvement would be to apply SMOTE only after partitioning.

6. **Momentum should complement, not dominate.** At $\alpha = 0.9$, the momentum term overwhelmed the gradient signal, preventing learning. Reducing to $\alpha = 0.5$ provided enough smoothing without drowning out the gradient.
