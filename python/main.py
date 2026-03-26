#!/usr/bin/env python
import numpy as np
import os
import sys

from utils import piechart, Partition, train, loadWeights, runPredictions, exportPredictions, learningcurve
from network import Epoch
from config import NetworkA, NetworkB

def main():
    print("Loading dataset...")
    DATASET     = np.loadtxt(os.path.join("dataset", "data.csv"), delimiter=",")
    DATALABELS  = np.loadtxt(os.path.join("dataset", "data_labels.csv"), delimiter=",", dtype=int)
    TESTSET     = np.loadtxt(os.path.join("dataset", "test_set.csv"), delimiter=",")
    
    CLASSES = int(np.max(DATALABELS))
    LABELS = np.zeros((len(DATALABELS), CLASSES), dtype=float)
    for _i, label in enumerate(DATALABELS):
        LABELS[_i, int(label) - 1] = 1.0

    piechart(DATALABELS, "Class Distribution Original Dataset")
    
    print("Running SMOTE...")
    from imblearn.over_sampling import SMOTE
    X_balanced, y_balanced = SMOTE(random_state=50).fit_resample(DATASET, DATALABELS)
    piechart(y_balanced, "Class Distribution after SMOTE")
    
    print("Partitioning the dataset...")
    split = Partition(X_balanced, y_balanced)
    split.printdetails()
    
    # ─── Training Configurations ──────────────────────────────────────────────
    
    epochA = Epoch(split, NetworkA)
    epochA.label = "Network A - Tanh (eta=0.85, alpha=0.9, size=8)"
    train(epochA, "Learning curve: Network A")
    epochA.exportAll("export/networkA/")

    # Improvement 1
    NetworkA['eta'] = 0.1
    NetworkA['alpha'] = 0.5
    NetworkA['size'] = 12
    epochA_improv = Epoch(split, NetworkA)
    epochA_improv.label = "Network A - Tanh (eta=0.1, alpha=0.5, size=12)"
    train(epochA_improv, "Learning curve: Network A improved")
    epochA_improv.exportAll("export/networkA_improv/")

    # Improvement 2
    NetworkA['eta'] = 0.05
    epochA_improv2 = Epoch(split, NetworkA)
    epochA_improv2.label = "Network A - Tanh (eta=0.05, alpha=0.5, size=12)"
    train(epochA_improv2, "Learning curve: Network A improved 2")
    epochA_improv2.exportAll("export/networkA_improv2/")

    # Improvement 2 Fast
    NetworkA['alpha'] = 0.9
    epochA_improv2_fast = Epoch(split, NetworkA)
    epochA_improv2_fast.label = "Network A - Tanh (eta=0.05, alpha=0.9, size=12)"
    train(epochA_improv2_fast, "Learning curve: Network A improved 2 (Fast)")
    epochA_improv2_fast.exportAll("export/networkA_improv2_fast/")

    # Improvement 2 Small
    NetworkA['size'] = 5
    epochA_improv2_small = Epoch(split, NetworkA)
    epochA_improv2_small.label = "Network A - Tanh (eta=0.05, alpha=0.9, size=5)"
    train(epochA_improv2_small, "Learning curve: Network A improved 2 (Small)")
    epochA_improv2_small.exportAll("export/networkA_improv2_small/")
    
    allepochA = [epochA, epochA_improv, epochA_improv2, epochA_improv2_fast, epochA_improv2_small]
    learningcurve(allepochA, "Learning Curves of Network A compared")

    # Network B
    epochB = Epoch(split, NetworkB)
    epochB.label = "Network B - ReLU (eta=0.85, alpha=0.9, size=8)"
    train(epochB, "Learning Curve: Network B")
    epochB.exportAll("export/networkB/")

    # Network B Improvement 1
    NetworkB['eta'] = 0.1
    NetworkB['alpha'] = 0.5
    NetworkB['size'] = 12
    epochB_improv = Epoch(split, NetworkB)
    epochB_improv.label = "Network B - ReLU (eta=0.1, alpha=0.5, size=12)"
    train(epochB_improv, "Learning curve: Network B improved")
    epochB_improv.exportAll("export/networkB_improv/")

    # Network B Improvement 2
    NetworkB['eta'] = 0.05
    epochB_improv2 = Epoch(split, NetworkB)
    epochB_improv2.label = "Network B - ReLU (eta=0.05, alpha=0.5, size=12)"
    train(epochB_improv2, "Learning curve: Network B improved 2")
    epochB_improv2.exportAll("export/networkB_improv2/")

    # Network B Improvement 2 Fast
    NetworkB['alpha'] = 0.9
    epochB_improv2_fast = Epoch(split, NetworkB)
    epochB_improv2_fast.label = "Network B - ReLU (eta=0.05, alpha=0.9, size=12)"
    train(epochB_improv2_fast, "Learning curve: Network B improved 2 (Fast)")
    epochB_improv2_fast.exportAll("export/networkB_improv2_fast/")

    # Network B Improvement 2 Small
    NetworkB['size'] = 5
    epochB_improv2_small = Epoch(split, NetworkB)
    epochB_improv2_small.label = "Network B - ReLU (eta=0.05, alpha=0.9, size=5)"
    train(epochB_improv2_small, "Learning curve: Network B improved 2 (Small)")
    epochB_improv2_small.exportAll("export/networkB_improv2_small/")
    
    allepochB = [epochB, epochB_improv, epochB_improv2, epochB_improv2_fast, epochB_improv2_small]
    learningcurve(allepochB, "Learning Curves of Network B compared")
    
    print("\nTraining completed for all networks.")
    
    # ─── Exporting Best Models ───────────────────────────────────────────────
    
    epochA_improv2_fast.exportWeights("modelA/")
    epochB_improv2.exportWeights("modelB/")
    
    WEIGHTSA = os.path.join("modelA", "trained_weights.csv")
    WEIGHTSB = os.path.join("modelB", "trained_weights.csv")
    MODELA = loadWeights(WEIGHTSA)
    MODELB = loadWeights(WEIGHTSB)
    
    predictions_tanh = runPredictions(MODELA, TESTSET, NetworkA)
    predictions_relu = runPredictions(MODELB, TESTSET, NetworkB)
    exportPredictions(predictions_tanh, filename="predictions_for_test_tanh.csv")
    exportPredictions(predictions_relu, filename="predictions_for_test_leakyrelu.csv")
    
    print("\n✅ All training, scoring, and prediction exports complete.")

if __name__ == "__main__":
    main()
