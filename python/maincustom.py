#!/usr/bin/env python
"""
Implementing the Backpropagation Algorithm — Custom Run
=======================================================
A standalone script to execute a custom neural network configuration (NetworkC)
without running the full sweep of baseline networks.
"""

import numpy as np
import os

# ─── 1. Loading the Dataset ──────────────────────────────────────────────────
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

from utils import piechart, Partition, train, loadWeights, runPredictions, exportPredictions
from network import Epoch

# ─── 2. Dataset Distribution ─────────────────────────────────────────────────
piechart(DATALABELS, "Class Distribution Original Dataset") 

# ─── 3. SMOTE ────────────────────────────────────────────────────────────────
X = DATASET
y = DATALABELS

from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE(random_state=50).fit_resample(X, y)
piechart(y_balanced, "Class Distribution after SMOTE")

# ─── 4. Partitioning the Dataset ─────────────────────────────────────────────
split = Partition(X_balanced, y_balanced)
split.printdetails()

# ─── 5. Hyperparameters Setup ────────────────────────────────────────────────
from config import NetworkC

# Feel free to dynamically modify NetworkC here before passing it to the Epoch
NetworkC['eta'] = 0.1
NetworkC['alpha'] = 0.5
NetworkC['size'] = 10

# ─── 6. Training: Custom Network (NetworkC) ──────────────────────────────────
epochC = Epoch(split, NetworkC)
te, ve = train(epochC, "Learning Curve Network Custom")
epochC.exportAll("export/networkCustom/")

# ─── 7. Exporting Best Models ────────────────────────────────────────────────
epochC.exportWeights("modelCustom/")

# ─── 8. Running and Exporting Predictions ────────────────────────────────────
print(f"TESTSET : {TESTSET.shape}")

WEIGHTSC = os.path.join("modelCustom", "trained_weights.csv")
MODELC = loadWeights(WEIGHTSC)

predictions_custom = runPredictions(MODELC, TESTSET, NetworkC)

exportPredictions(predictions_custom, filename="predictions_for_test_custom.csv")

print("\n✅ Custom training, scoring, and prediction exports complete.")
