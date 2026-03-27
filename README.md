# Implementing the Backpropagation Algorithm

A Multilayer Perceptron (MLP) Neural Network trained using the Backpropagation Algorithm on a challenging 8-class dataset. This project explores the full pipeline — from data preprocessing and class balancing, through architecture design and mathematical formulation, to iterative hyperparameter tuning and model evaluation.

---

## Project Structure

The project has been refactored into a modern, modular Python architecture located in `python/`, containing the core components (`main.py`, `network.py`, `activations.py`, etc.). The original Jupyter notebook (`main.ipynb`) and its outputs are archived in `submission/`.

```text
mbtiongson/neural-network-training/
├── LICENSE
├── README.md
├── .gitignore
├── python/
│   ├── main.py
│   ├── activations.py
│   ├── network.py
│   ├── utils.py
│   ├── checkscores.py
│   ├── config.py
│   ├── dataset/
│   ├── figures/
│   ├── export/
│   ├── modelA/
│   ├── modelB/
│   └── predictions/
└── submission/
    ├── main.ipynb
    ├── checkscores.py
    ├── dataset/
    ├── export/
    ├── final/
    ├── modelA/
    ├── modelB/
    ├── predictions/
    └── combined_scores.csv
```

### Running the Python Version for Deployment

For a production or deployment environment, the refactored modular python structural format is recommended. Ensure the library requirements are satisfied first:

```bash
cd python/
pip install -r requirements.txt
```

To run the pipeline and output standard prediction artifacts:

```bash
cd python/
python main.py           # Train all networks and export predictions to predictions/ directory
python checkscores.py    # Aggregate and rank scores from export/
```

### Experimenting with Custom Configurations

If you would like to run a single configuration without training the full suite of networks, a custom run feature is provided:

1. Open `python/config.py` and edit the `NetworkC` dictionary (you can tweak hyperparameters, learning rates, or activation functions).
2. Inside `python`, run the custom script:

   ```bash
   python maincustom.py
   ```

3. The model weight outputs will be exported to `modelCustom/trained_weights.csv`, the plots will be labeled as "Custom", and the predictions will be saved to `predictions/predictions_for_test_custom.csv`.

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
| Output activation | Logistic ($a=1.0$) | Logistic ($a=2.0$) |
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

The hyperparameters were adjusted based on the observation that convergence was failing. Lowering the learning rate ($\eta$) to 0.1 or 0.05 was the primary factor enabling successful training.

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

The results improved dramatically. Errors converged steadily, and the validation error aligned well with the training error. A hidden size of **12** was found to be more reliable across configurations.

Later experiments showed that **higher momentum** could improve speed *after* the learning rate had been stabilized. For example, `networkA_improv2_fast` ($\eta=0.1, \alpha=0.9$) converged much faster, reaching its target error threshold by the 44th epoch.

### Training: Network B

#### Default Parameters — First Attempt

Network B, using Leaky ReLU for hidden layers, was trained with the same default parameters (`eta=0.85`, `alpha=0.9`, `size=8`). The result was identical to Network A's failure — the errors did not change, and the F1 score was very low.

#### Improvements for Network B

The same tuning strategy that worked for Network A was applied, lowering the learning rate to 0.1 and increasing the hidden nodes to 12 (`networkB_improv2`).

```python
NetworkB['eta'] = 0.1    # decreasing the learning rate
NetworkB['alpha'] = 0.5  # decreasing the momentum
NetworkB['size'] = 12    # increasing hidden layer nodes
```

This produced the **best result of all configurations**, demonstrating Leaky ReLU's superior peak performance when hyperparameter settings are appropriate. As noted during testing, reducing the hidden layer to 5 weakened Leaky ReLU considerably, confirming that a size of 12 was indeed necessary for this architecture.

---

## Training Results

### F1 Score Rankings

| Rank | Network | Epoch | F1 Score | Time (s) | Time-Adjusted |
|------|---------|-------|----------|----------|---------------|
| 🥇 1 | **networkB_improv2** | 73 | **0.9886** | 96.9 | 70.8 |
| 🥈 2 | **networkB_improv2_fast** | 51 | **0.9853** | 87.4 | 44.6 |
| 🥉 3 | **networkA_improv2_fast** | 44 | **0.9835** | 67.5 | 29.7 |
| 4 | networkA_improv2_small | 74 | 0.9811 | 42.4 | 31.3 |
| 5 | networkA_improv | 56 | 0.9801 | 52.2 | 29.3 |
| 6 | networkA_improv2 | 74 | 0.9761 | 59.3 | 43.9 |
| 7 | networkB_improv | 71 | 0.9613 | 74.4 | 52.8 |

### Full Scores

| Rank | Network | Epoch | F1 Score | MCC | Accuracy | Precision | Recall |
|------|---------|-------|----------|-----|----------|-----------|--------|
| 1 | **networkB_improv2** | 73 | **0.9886** | 0.9871 | 0.9888 | 0.9880 | 0.9893 |
| 2 | **networkB_improv2_fast** | 51 | **0.9853** | 0.9829 | 0.9850 | 0.9856 | 0.9852 |
| 3 | **networkA_improv2_fast** | 44 | **0.9835** | 0.9801 | 0.9825 | 0.9840 | 0.9838 |
| 4 | networkA_improv2_small | 74 | 0.9811 | 0.9786 | 0.9813 | 0.9812 | 0.9814 |
| 5 | networkA_improv | 56 | 0.9801 | 0.9772 | 0.9800 | 0.9804 | 0.9807 |
| 6 | networkA_improv2 | 74 | 0.9761 | 0.9731 | 0.9763 | 0.9773 | 0.9768 |
| 7 | networkB_improv | 71 | 0.9613 | 0.9548 | 0.9600 | 0.9647 | 0.9614 |

**Network B (`networkB_improv2`)** with Leaky ReLU is the clear winner with a **0.9886 F1 score**. Its trained weights were exported to `modelA/trained_weights.csv` as the primary model for unseen test data.

**Network A (`networkA_improv2_fast`)** with Tanh came in second functionally among the converging models for having the best speed-performance tradeoff (**0.9835 F1** at epoch 44). It loads faster and scales better on longer epoch iterations. Its weights were exported to `modelB/trained_weights.csv` as the backup model.

The bottom configurations (using default hyperparameters and early small hidden layers like `networkB_improv2_small`) achieved near-zero F1 scores (DNF), confirming they failed to converge.

---

## Loading the Model

Trained weights are loaded from CSV files using the `loadWeights` function, which reconstructs the weight matrices for each layer.

## Running and Exporting Predictions

The `runPredictions` function takes the loaded weights and the test set, reconstructs the network layers, and performs a forward pass for each test sample. Predictions are determined by taking the `argmax` of the output layer activations.

The best two models were used to generate predictions:

- **Network B** (Leaky ReLU) → `predictions/predictions_for_test_leakyrelu.csv`
- **Network A** (Tanh) → `predictions/predictions_for_test_tanh.csv`

---

## Conclusion and Recommendations

This study evaluated ten Multilayer Perceptron configurations across two activation-function families, Tanh and Leaky ReLU, by varying the learning rate ($\eta$), momentum constant ($\alpha$), and hidden-layer size. The conclusions below follow the ranked results and selected models reported in the **Training Results** section.

## Conclusions

1. **The default configuration was not usable.** The baseline `networkA` and `networkB` runs both failed to converge and ranked last, with macro F1 scores of **0.02925** and **0.02503** respectively. This confirms that the original $\eta = 0.85$ and $\alpha = 0.9$ setting was too aggressive for this task.

2. **Lowering the learning rate was the main factor that enabled successful training.** Once $\eta$ was reduced to **0.1** or **0.05**, both network families produced strong models. The top overall result came from **`networkB_improv2`**, which achieved a macro F1 score of **0.98861** and reached the target error threshold at **epoch 73**.

3. **Higher momentum improved speed after the learning rate had been stabilized.** This is most visible in **`networkA_improv2_fast`**, which reached the error target by **epoch 44** with a macro F1 score of **0.98352** and the best time-adjusted training cost among converging models. For Network B, **`networkB_improv2_fast`** also reduced convergence time to **epoch 51** while maintaining a strong macro F1 score of **0.98530**.

4. **A hidden-layer size of 12 was more reliable than 5.** Reducing the hidden size to 5 remained acceptable for Tanh (`networkA_improv2_small`, macro F1 **0.98108**), but it clearly weakened Leaky ReLU (`networkB_improv2_small`), which did not finish successfully and dropped to a macro F1 score of **0.55645**.

5. **Leaky ReLU delivered the highest peak accuracy, while Tanh offered the best efficiency-accuracy balance.** The two highest-ranked models overall were the Leaky ReLU variants `networkB_improv2` and `networkB_improv2_fast`, but the Training Results section also shows that **`networkA_improv2_fast`** is the strongest time-efficient choice among the converging models.

## Recommendations

- **For the highest predictive performance**, use **`networkB_improv2`** as the primary model because it achieved the best overall macro F1 score (**0.98861**).

- **For the strongest speed-performance tradeoff**, use **`networkA_improv2_fast`**. It converged by **epoch 44**, achieved a macro F1 score of **0.98352**, and had the lowest time-adjusted training cost among the successful runs.

- **For model selection**, retain the chosen pair already identified in the Training Results section: **`networkA_improv2_fast`** for Tanh and **`networkB_improv2`** for Leaky ReLU.

- **For future work**, implement formal early stopping based on validation error and apply SMOTE only after the train-validation split so the validation set remains fully independent.
