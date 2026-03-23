import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os, csv, time

from network import HiddenLayer, OutputLayer, Epoch

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# ─── Dataset Distribution Visualization ───────────────────────────────────────

def piechart(datalabels, title='Class Distribution'): #use trainingset labels
    classcounts = np.bincount(datalabels, minlength=9)[1:]
    labels = [f"Class {i}" for i in range(1, len(classcounts) + 1)]

    plt.figure(figsize=(10, 8))
    plt.pie(classcounts, autopct=lambda pct: f'{int(pct/100.*sum(classcounts))}\n({pct:.1f}%)', startangle=90)
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(title)
    plt.axis('equal')

    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    safetitle = title.replace(" ", "_").replace(":", "").replace("/", "_")
    filepath = os.path.join(FIGURES_DIR, f"{safetitle}.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"Saved figure → {filepath}")
    plt.close()

# ─── Partitioning the Dataset ─────────────────────────────────────────────────

class Partition: #X is the dataset, y is the datalabels
    def __init__(self, X, y, valsize=800, outputdir="export", randomstate=50):
        os.makedirs(outputdir, exist_ok=True)
        rng = np.random.default_rng(randomstate)

        indices = rng.permutation(len(X))
        validx, trainidx = indices[:valsize], indices[valsize:]

        Xtrain, ytrain = X[trainidx], y[trainidx]
        Xval, yval = X[validx], y[validx]

        self.classes = np.unique(y)
        self.outputdir = outputdir
        self.validationset = Xval
        self.validationlabels = yval
        self.trainingset = Xtrain
        self.traininglabels = ytrain

        self.exportcsv(self.trainingset, "training_set.csv")
        self.exportcsv(self.traininglabels, "training_labels.csv")
        self.exportcsv(self.validationset, "validation_set.csv")
        self.exportcsv(self.validationlabels, "validation_labels.csv")

    def printdetails(self):
        print("Training Set Details")
        print(f"  Shape       : {self.trainingset.shape}")
        print(f"  Label shape : {self.traininglabels.shape}")
        print(f"  Feature min : {self.trainingset.min():.6f}")
        print(f"  Feature max : {self.trainingset.max():.6f}")
        print(f"  Feature mean: {self.trainingset.mean():.6f}")
        piechart(self.traininglabels, "Class Distribution of Training Set")
        print(f"\nValidation Set Details")
        print(f"  Shape       : {self.validationset.shape}")
        print(f"  Label shape : {self.validationlabels.shape}")
        piechart(self.validationlabels, "Class Distribution of Validation Set")

    #all export functions after split
    def exportcsv(self, data, filename):
        path = os.path.join(self.outputdir, filename)
        fmt = "%d" if data.ndim == 1 else "%g"
        np.savetxt(path, data, delimiter=",", fmt=fmt)
        print(f"Saved → {path}  shape: {data.shape}")

# ─── Mini-batch Generator ─────────────────────────────────────────────────────

def minibatch(trainingset, traininglabels, batch_size=8):
    N = len(trainingset)
    indices = np.arange(N)
    for start in range(0, N, batch_size):
        i = indices[start : start + batch_size]
        yield trainingset[i], traininglabels[i]

# ─── Validation Error ─────────────────────────────────────────────────────────

def computeValError(epoch):
    total_mse = 0.0
    n = len(epoch.test_set)
    for x, d in epoch.test_set:
        x_biased = np.concatenate(([1.0], np.asarray(x, dtype=float)))
        phi_i = epoch.hiddenlayer_i.forward(x_biased)
        phi_j = epoch.hiddenlayer_j.forward(phi_i)
        epoch.outputlayer_k.forward(phi_j)
        n_out = epoch.outputlayer_k.w_old.shape[0]
        d_onehot = np.zeros(n_out)
        d_onehot[int(d) - 1] = 1.0
        epoch.outputlayer_k.computeError(d_onehot)
        total_mse += epoch.outputlayer_k.mse
    return total_mse / n if n > 0 else 0.0

# ─── Learning Curve Plot ──────────────────────────────────────────────────────

def learningcurve(train_errors, val_errors, networkname="Learning Curve Training vs Validation Error"):
    epochs = range(1, len(train_errors) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_errors, label="Training Error (MSE)")
    plt.plot(epochs, val_errors,   label="Validation Error (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(networkname)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    safetitle = networkname.replace(" ", "_").replace(":", "").replace("/", "_")
    filepath = os.path.join(FIGURES_DIR, f"{safetitle}.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"Saved figure → {filepath}")
    plt.close()

# ─── Training Function ────────────────────────────────────────────────────────

def train(epoch, networkname="Learning Curve Training vs Validation Error", epochs=100):
    """Train the network for the given number of epochs.
    Returns (train_errors, val_errors) lists for use in consolidated_learningcurve."""
    batch_size   = epoch.config.get('batch_size', 8)
    train_errors = []
    val_errors   = []
    totaltime = 0
    for ep in range(epochs):
        start = time.time()
        epoch_error = 0.0
        n_batches = 0
        for xbatch, dbatch in minibatch(epoch.trainingset, epoch.traininglabels, batch_size):
            epoch_error += epoch.run_batch(xbatch, dbatch)
            n_batches += 1
        epoch_error /= max(n_batches, 1)
        val_error = computeValError(epoch)
        train_errors.append(epoch_error)
        val_errors.append(val_error)
        elapsed = time.time() - start
        totaltime += elapsed
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1:>4}  Train: {epoch_error:.5f}  Val: {val_error:.5f}  Time: {elapsed:.3f}s")
    epoch.totaltime = totaltime
    epoch.Scores()
    epoch.printScores()
    print(f"Total training time: {totaltime:.2f}s")
    learningcurve(train_errors, val_errors, networkname)
    return train_errors, val_errors

# ─── Consolidated Learning Curve (all networks) ───────────────────────────────

def consolidated_learningcurve(all_results):
    """Plot all network learning curves on a single figure.
    all_results: list of tuples (name, train_errors, val_errors)"""
    plt.figure(figsize=(14, 7))
    for name, train_errors, val_errors in all_results:
        epochs = range(1, len(train_errors) + 1)
        plt.plot(epochs, train_errors, label=f"{name} (Train)")
        plt.plot(epochs, val_errors, linestyle='--', label=f"{name} (Val)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Consolidated Learning Curves — All Networks")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, "Consolidated_Learning_Curves.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"Saved consolidated figure → {filepath}")
    plt.close()

# ─── Weight Loading ───────────────────────────────────────────────────────────

def loadWeights(path):
    blocks = {'Wi': [], 'Wj': [], 'Wk': []}
    current = None
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip() == '':
                continue
            tag = row[0].strip()
            if tag in blocks:
                current = tag
                continue
            if current is not None:
                blocks[str(current)].append([float(v) for v in row])
    return (np.array(blocks['Wi']),
            np.array(blocks['Wj']),
            np.array(blocks['Wk']))

# ─── Prediction ───────────────────────────────────────────────────────────────

def runPredictions(model, testset, cfg):
    if isinstance(model, tuple):
        Wi, Wj, Wk = model
        layerI = HiddenLayer(cfg['methods'][0], Wi, cfg, size=Wi.shape[0])
        layerJ = HiddenLayer(cfg['methods'][1], Wj, cfg, size=Wj.shape[0])
        layerK = OutputLayer(cfg['methods'][2], Wk, cfg)
    else:
        layerI = model.hiddenlayer_i
        layerJ = model.hiddenlayer_j
        layerK = model.outputlayer_k

    predictions = []
    for x in testset:
        xb   = np.concatenate(([1.0], np.asarray(x, dtype=float)))
        phiI = layerI.forward(xb)
        phiJ = layerJ.forward(phiI)
        layerK.forward(phiJ)
        label = int(np.argmax(layerK.o)) + 1
        predictions.append(label)
    return predictions

def exportPredictions(predictions, filename="predictions_for_test_networkA.csv", outputdir="predictions"):
    os.makedirs(outputdir, exist_ok=True)
    filepath = os.path.join(outputdir, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for p in predictions:
            writer.writerow([p])
    print(f"Saved {len(predictions)} predictions → {filepath}")
