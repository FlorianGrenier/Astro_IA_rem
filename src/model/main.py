# main_simple.py
import os, glob, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import SpectralBatchDataset



from model import SpectrumCNN
from train import train_model, test_model


from result import (
    plot_roc_auc, plot_confusion_matrices,
    plot_precision_recall, plot_learning_curves,
    plot_ap_per_class, plot_global_confusion_matrix,
    plot_global_confusion_matrix, plot_learning_curves,
)

from sklearn.metrics import average_precision_score




import torch.multiprocessing as mp




SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect(); 
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"[DEVICE] {device}")

# transforms_alma.py
import torch

@torch.no_grad()
def triple_normalize_1d(x, alpha: float = 1.5, eps: float = 1e-8):
   
    v = x.squeeze(0).float()            # (N,)
    m = v.abs().max().clamp_min(eps)
    ch0 = v / m                         # max-norm
    ch1 = torch.tanh(alpha * v)
    ch1 = ch1 / ch1.abs().max().clamp_min(eps)
    ch2 = ch0 ** 3
    return torch.stack([ch0, ch1, ch2], dim=0)   # (3, N)

@torch.no_grad()
def random_jitter_roll(x, max_shift: int = 2):
    
    if max_shift <= 0:
        return x
    s = int(torch.randint(-max_shift, max_shift + 1, ()).item())
    if s == 0:
        return x
    return torch.roll(x, shifts=s, dims=-1)

@torch.no_grad()
def alma_transform_from_numpy(x_np, jitter: int = 2):
   
    x = torch.from_numpy(x_np).float()         
    x3 = triple_normalize_1d(x)                 
    x3 = random_jitter_roll(x3, max_shift=jitter)
    return x3


spectra_files = sorted(glob.glob("data_generate/synthetic_spectra_polluted_batch_*.npy"))
labels_files  = sorted(glob.glob("data_generate/synthetic_labels_corrected_batch_*.npy"))
assert spectra_files and labels_files, "Aucun fichier trouvé dans data_generate/"

names_path = "data_generate/molecule_names.txt" if os.path.exists("data_generate/molecule_names.txt") else "molecule_names.txt"
with open(names_path, "r") as f:
    molecule_names = [line.strip() for line in f if line.strip()]
num_classes = len(molecule_names)
print(f"[INFO] {num_classes} molécules.")

freqs_path = "data_generate/frequencies.npy"
assert os.path.exists(freqs_path), "frequencies.npy manquant"
grid_freqs = np.load(freqs_path)
NCH = len(grid_freqs)

print(f"[INFO] NCH = {NCH} (longueur des spectres)")



ds = SpectralBatchDataset(spectra_files, labels_files, transform=alma_transform_from_numpy)
ds_raw = SpectralBatchDataset(spectra_files, labels_files, transform=None)

ALL = len(ds)
idx_all = np.arange(ALL)
train_idx, tmp_idx = train_test_split(idx_all, test_size=0.2, random_state=SEED, shuffle=True)
val_idx,   test_idx = train_test_split(tmp_idx,   test_size=0.5, random_state=SEED, shuffle=True)
print(f"[SPLIT] train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")


# === Fonctions utilitaires pour stats ===
def collect_labels(subset, batch_size=1024, max_batches=None):
   
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    ys = []
    for b, (xb, yb) in enumerate(loader):
        ys.append(yb.numpy())
        if max_batches is not None and (b+1) >= max_batches:
            break
    return np.concatenate(ys, axis=0) if ys else np.zeros((0, num_classes), dtype=np.float32)


def collect_spectra_sample(subset, k=2000, batch_size=1024):
   
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    xs = []
    for xb, yb in loader:
        # ds_raw renvoie x: (B,1,NCH) torch.float32
        x_np = xb.numpy()[:, 0, :]  # (B, NCH)
        xs.append(x_np.astype(np.float32))
        if sum(len(a) for a in xs) >= k:
            break
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, NCH), dtype=np.float32)
    return X[:k]

def hash_batch(X):
    return [hashlib.sha1(x.tobytes()).hexdigest() for x in X]



import hashlib, numpy as np


# Construire les Subset RAW pour checks
train_ds_raw = Subset(ds_raw, train_idx)
val_ds_raw   = Subset(ds_raw, val_idx)
test_ds_raw  = Subset(ds_raw, test_idx)

# ---- Cardinalité moyenne (nb de molécules par spectre) ----
y_train = collect_labels(train_ds_raw)
y_val   = collect_labels(val_ds_raw)
print("Card. moyenne TRAIN:", float(y_train.sum(1).mean()) if len(y_train) else "N/A")
print("Card. moyenne VAL  :", float(y_val.sum(1).mean())   if len(y_val)   else "N/A")


# ---- Doublons inter-splits (hash sur un échantillon) ----
X_train_sample = collect_spectra_sample(train_ds_raw, k=2000)
X_val_sample   = collect_spectra_sample(val_ds_raw,   k=2000)
X_test_sample  = collect_spectra_sample(test_ds_raw,  k=2000)

Htr = set(hash_batch(X_train_sample))
Hval= set(hash_batch(X_val_sample))
Hte = set(hash_batch(X_test_sample))
print("Dup TRAIN∩VAL:", len(Htr & Hval))
print("Dup TRAIN∩TEST:", len(Htr & Hte))
print("Dup VAL∩TEST:", len(Hval & Hte))


# ---- Baseline mAP micro (aléatoire) sur VAL ----
if len(y_val):
    rand_probs = np.random.rand(*y_val.shape)
    print("mAP micro baseline (random):", average_precision_score(y_val.ravel(), rand_probs.ravel()))
else:
    print("mAP micro baseline (random): N/A (val vide)")

ds = SpectralBatchDataset(spectra_files, labels_files, transform=alma_transform_from_numpy)

train_ds = Subset(ds, train_idx)
val_ds   = Subset(ds, val_idx)
test_ds  = Subset(ds, test_idx)

BATCH = 32

num_workers = min(8, os.cpu_count() if os.cpu_count() is not None else 1)
print(f"[INFO] Using num_workers={num_workers} for DataLoaders")

use_gpu = torch.cuda.is_available()
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=num_workers, pin_memory=use_gpu)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=num_workers, pin_memory=use_gpu)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=num_workers, pin_memory=use_gpu)



# ============== Modèle, loss, optim, scheduler ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect(); torch.cuda.empty_cache()
print(f"[DEVICE] {device}")

model = SpectrumCNN(num_classes=8, in_ch=3, c=64, p_drop=0.25)
model = model.to(device)
print(model)

def estimate_pos_weight(loader, num_classes):
    pos = torch.zeros(num_classes, dtype=torch.float64)
    tot = 0
    for _, yb in loader:
        pos += yb.sum(dim=0).double()
        tot += yb.shape[0]
    neg = tot - pos
    pw = (neg / pos.clamp_min(1)).clamp(max=50.0)  # >=1 si pos>0
    return pw.float()



with torch.no_grad():
    pos_weight = estimate_pos_weight(train_loader, num_classes).to(device)

criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


bce = torch.nn.BCEWithLogitsLoss()


criterion = nn.BCEWithLogitsLoss()                # multi-label
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# print(f"[INFO] pos_weight: {pos_weight.cpu().numpy()}")

# # ============== Entraînement ==============
save_path = "best_model_simple0.pth"
history, best_thr_per_class = train_model(
    model, train_loader, val_loader, optimizer, criterion, device,
    epochs=50, scheduler=scheduler, save_path="best_model_simple.pth",
    patience=6, log_train_metrics=True, thr_method="quantiles",
    recompute_thr_every=5, freeze_thr_after=10
)

# ============== Test ==============
probs, targets, _ = test_model(model, test_loader, device, path=save_path)
print("[DONE] Évaluation test terminée.")


# === Plots résultats ===
plot_ap_per_class(
    probs,
    targets,
    class_names=molecule_names,
    top_k=8,   
    save_path="ap_per_class_simple0.png"
)
plot_global_confusion_matrix(probs, targets,
                                threshold=0.5, save_path="global_confusion_simple0.png")
plot_learning_curves(history, save_path="learning_curves_simple0.png")
plot_roc_auc(probs, targets, class_names=molecule_names,
             save_path="roc_auc_simple0.png")

plot_precision_recall(probs, targets, class_names=molecule_names,
                      save_path="precision_recall_simple0.png")
plot_confusion_matrices(probs, targets, class_names=molecule_names,
                        threshold=0.5, limit_classes=10,
                        save_path="confusion_matrices_simple0.png")
print("[DONE] Plots saved.")
# ============== Sauvegarde modèle final ==============
final_model_path = "final_model_simple0.pth"
torch.save(model.state_dict(), final_model_path)
print(f"[SAVED] Modèle final sauvegardé dans {final_model_path}")
