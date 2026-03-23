#!/usr/bin/env python
"""
Implementing the Backpropagation Algorithm
==========================================
A Multilayer Perceptron (MLP) Neural Network trained using the
Backpropagation Algorithm on a challenging 8-class dataset.

This script is the consolidated, pruned Python equivalent of main.ipynb.
"""

import numpy as np
import os

# ─── 1. Loading the Dataset ──────────────────────────────────────────────────
# The dataset is loaded from CSV files in the dataset/ directory.
# Features are dense numerical vectors with 2052 dimensions, and
# labels range from 1 to 8 (8 classes).

DATASET     = np.loadtxt(os.path.join("dataset", "data.csv"), delimiter=",")
DATALABELS  = np.loadtxt(os.path.join("dataset", "data_labels.csv"), delimiter=",", dtype=int)
TESTSET     = np.loadtxt(os.path.join("dataset", "test_set.csv"), delimiter=",")

print(f"DATASET : {DATASET.shape}")
print(f"DATALABELS : {DATALABELS.shape}  classes: {np.unique(DATALABELS)}")
print(f"TESTSET : {TESTSET.shape}")

CLASSES = int(np.max(DATALABELS)) #1,2,3,4,5,6,7,8
LABELS = np.zeros((len(DATALABELS), CLASSES), dtype=float) #what class is the data
for _i, label in enumerate(DATALABELS):
    LABELS[_i, int(label) - 1] = 1.0

print(f"LABELS   : {LABELS.shape}")
print(f"Sample   : label {DATALABELS[0]} → {LABELS[0]}")

# ─── 2. Dataset Distribution ─────────────────────────────────────────────────
# Visualize class distribution before and after SMOTE.

from utils import piechart, Partition, train, loadWeights, runPredictions, exportPredictions, consolidated_learningcurve
from network import Epoch

piechart(DATALABELS, "Class Distribution Original Dataset") #the original

# ─── 3. SMOTE (Synthetic Minority Over-sampling Technique) ───────────────────
# SMOTE generates synthetic samples for minority classes to balance
# the class distribution.

X = DATASET
y = DATALABELS

from imblearn.over_sampling import SMOTE

X_balanced, y_balanced = SMOTE(random_state=50).fit_resample(X, y)

piechart(y_balanced, "Class Distribution after SMOTE")

# ─── 4. Partitioning the Dataset ─────────────────────────────────────────────
# Validation set: 800 samples.  Training set: remaining samples.

split = Partition(X_balanced, y_balanced)
split.printdetails()

# ─── 5. Hyperparameters Setup ────────────────────────────────────────────────
# Two network configurations: Network A (Tanh) and Network B (Leaky ReLU).

NetworkA = {
        'methods'    : [1, 1, 0],  # [i, j, k] — 0 logistic, 1 tanh, 2 relu
        'a_l'        : 1.0,        # logistic slope
        'a_tanh'     : 1.716,      # tanh a
        'b_tanh'     : 0.66666,    # tanh b
        'a_relu'     : 0.01,       # leaky relu gamma
        'eta'        : 0.85,       # learning rate
        'alpha'      : 0.9,        # momentum constant
        'size'       : 8,          # hidden layer size
        'batch_size' : 8,          # mini-batch size
    }
NetworkB = {
        'methods'    : [2, 2, 0],  # [i, j, k] — 0 logistic, 1 tanh, 2 relu
        'a_l'        : 1.0,        # logistic slope
        'a_tanh'     : 1.716,      # tanh a
        'b_tanh'     : 0.66666,    # tanh b
        'a_relu'     : 0.01,       # leaky relu gamma
        'eta'        : 0.85,        # learning rate
        'alpha'      : 0.9,        # momentum constant
        'size'       : 8,          # hidden layer size
        'batch_size' : 8,          # mini-batch size
    }

# Collector for consolidated learning curve
all_results = []

# ─── 6. Training: Network A (default) ────────────────────────────────────────
# Default config: eta=0.85, alpha=0.9, size=8.
# Error remained flat — the learning rate was too aggressive.

epochA = Epoch(split, NetworkA)
te, ve = train(epochA, "Learning curve Network A")
epochA.exportAll("export/networkA/")
all_results.append(("Network A", te, ve))

# ─── 7. Improvements for Network A ───────────────────────────────────────────
# Lowered eta from 0.85 → 0.1, alpha from 0.9 → 0.5, size from 8 → 12.

NetworkA['eta'] = 0.1 #lowering the learning rate
NetworkA['alpha'] = 0.5 #lowering the momentum
NetworkA['size'] = 12 #increasing the number of hidden layers

epochA_improv = Epoch(split, NetworkA)
te, ve = train(epochA_improv, "Learning curve Network A improved")
epochA_improv.exportAll("export/networkA_improv/")
all_results.append(("Network A improved", te, ve))

# ─── 8. Network A improved (Fast) ────────────────────────────────────────────
# Testing size=10 — smaller hidden layer, potentially comparable results.

NetworkA['alpha'] = 0.5 #increasing the momentum
NetworkA['size'] = 10 #decreasing the number of hidden layers back

epochA_improv_fast = Epoch(split, NetworkA)
te, ve = train(epochA_improv_fast, "Learning curve Network A improved (Fast)")
epochA_improv_fast.exportAll("export/networkA_improv_fast/")
all_results.append(("Network A improved (Fast)", te, ve))

# ─── 9. Training: Network B (default) ────────────────────────────────────────
# Network B uses Leaky ReLU for hidden layers and Logistic for output.

epochB = Epoch(split, NetworkB)
te, ve = train(epochB, "Learning Curve Network B")
epochB.exportAll("export/networkB/")
all_results.append(("Network B", te, ve))

# ─── 10. Improvements: Network B ─────────────────────────────────────────────
# Same tuning strategy: eta=0.1, alpha=0.5, size=10.

NetworkB['eta'] = 0.1 #decreasing the learning rate
NetworkB['alpha'] = 0.5 #decreasing the momentum
NetworkB['size'] = 10 #increasing hidden layer nodes

epochB_improv = Epoch(split, NetworkB)
te, ve = train(epochB_improv, "Learning curve Network B improved")
epochB_improv.exportAll("export/networkB_improv/")
all_results.append(("Network B improved", te, ve))

# ─── 11. Consolidated Learning Curve ─────────────────────────────────────────
# Plot all 5 training runs (6 curves counting duplicate default A) in one figure.

consolidated_learningcurve(all_results)

# ─── 12. Exporting Best Models ───────────────────────────────────────────────
# Network B (improved, Leaky ReLU) → modelA (best)
# Network A (improved, fast, Tanh) → modelB (backup)

epochB_improv.exportWeights("modelA/")
epochA_improv_fast.exportWeights("modelB/")

# ─── 13. Running and Exporting Predictions ───────────────────────────────────
# The best two models are loaded and used to predict the unseen test set.

#for sanity's sake, this is the parsed  TESTSET from test_set.csv
print(TESTSET)
print(f"TESTSET : {TESTSET.shape}")

WEIGHTSA = os.path.join("modelA", "trained_weights.csv") #tanh
WEIGHTSB = os.path.join("modelB", "trained_weights.csv") #leakyrelu

MODELA = loadWeights(WEIGHTSA)
MODELB = loadWeights(WEIGHTSB)

predictions_tanh = runPredictions(MODELA, TESTSET, NetworkA)
predictions_relu = runPredictions(MODELB, TESTSET, NetworkB)

#exporting predictions
exportPredictions(predictions_tanh, filename="predictions_for_test_tanh.csv")
exportPredictions(predictions_relu, filename="predictions_for_test_leakyrelu.csv")

print("\n✅ All training, scoring, and prediction exports complete.")
