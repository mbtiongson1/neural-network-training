#!/usr/bin/env python
import numpy as np
import os

from utils import piechart, Partition, train, loadWeights, runPredictions, exportPredictions, learningcurve
from network import Epoch
from config import NetworkC

def main():
    print("Loading dataset...")
    DATASET     = np.loadtxt(os.path.join("dataset", "data.csv"), delimiter=",")
    DATALABELS  = np.loadtxt(os.path.join("dataset", "data_labels.csv"), delimiter=",", dtype=int)
    TESTSET     = np.loadtxt(os.path.join("dataset", "test_set.csv"), delimiter=",")
    
    CLASSES = int(np.max(DATALABELS))
    LABELS = np.zeros((len(DATALABELS), CLASSES), dtype=float)
    for _i, label in enumerate(DATALABELS):
        LABELS[_i, int(label) - 1] = 1.0

    print("Running SMOTE...")
    from imblearn.over_sampling import SMOTE
    X_balanced, y_balanced = SMOTE(random_state=50).fit_resample(DATASET, DATALABELS)
    
    print("Partitioning the dataset...")
    split = Partition(X_balanced, y_balanced)
    split.printdetails()
    
    # ─── Training Configurations ──────────────────────────────────────────────
    
    epochC = Epoch(split, NetworkC)
    epochC.label = "Network Custom"
    train(epochC, "Learning curve: Network Custom")
    epochC.exportAll("export/networkCustom/")

    print("\nTraining completed for custom network.")
    
    # ─── Exporting Models ───────────────────────────────────────────────
    
    epochC.exportWeights("modelCustom/")
    
    WEIGHTSC = os.path.join("modelCustom", "trained_weights.csv")
    MODELC = loadWeights(WEIGHTSC)
    
    predictions_custom = runPredictions(MODELC, TESTSET, NetworkC)
    exportPredictions(predictions_custom, filename="predictions_for_test_custom.csv")
    
    print("\n✅ All training, scoring, and prediction exports complete.")

if __name__ == "__main__":
    main()
