import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, multilabel_confusion_matrix,
    precision_recall_curve, average_precision_score, confusion_matrix
)
import seaborn as sns

# =========================
# 1) ROC AUC par classe + micro
# =========================
def plot_roc_auc(probs, targets, class_names=None, save_path=None):
    n_classes = targets.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    # micro-average (sur tous les labels)
    fpr["micro"], tpr["micro"], _ = roc_curve(targets.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # par classe (skip si classe constante)
    valid_idx = []
    for i in range(n_classes):
        yi = targets[:, i]
        if yi.max() == yi.min():
            continue
        fpr[i], tpr[i], _ = roc_curve(yi, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        valid_idx.append(i)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro ROC (AUC={roc_auc["micro"]:.3f})',
             linestyle=':', linewidth=3)
    for i in valid_idx:
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(fpr[i], tpr[i], label=f'{label} (AUC={roc_auc[i]:.3f})', linewidth=1.2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC AUC (micro + classes valides)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.3); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); plt.close()
    else: plt.show()


# =========================
# 2) Courbes P-R par classe + AP
# =========================
def plot_precision_recall(probs, targets, class_names=None, save_path=None):
    n_classes = targets.shape[1]
    plt.figure(figsize=(10, 8))
    valid = 0
    for i in range(n_classes):
        yi = targets[:, i]
        if yi.max() == yi.min():
            continue
        precision, recall, _ = precision_recall_curve(yi, probs[:, i])
        ap = average_precision_score(yi, probs[:, i])
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(recall, precision, label=f'{label} (AP={ap:.3f})')
        valid += 1
    if valid == 0:
        plt.close()
        return
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall (classes valides)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.3); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); plt.close()
    else: plt.show()


# =========================
# 3) AP par classe (bar chart) — très utile
# =========================
def plot_ap_per_class(probs, targets, class_names=None, top_k=None, save_path=None):
    n_classes = targets.shape[1]
    aps = []
    names = []
    for i in range(n_classes):
        yi = targets[:, i]
        if yi.max() == yi.min():
            continue
        ap = average_precision_score(yi, probs[:, i])
        aps.append(ap); names.append(class_names[i] if class_names else f"C{i}")
    if not aps:
        return
    aps = np.array(aps); order = np.argsort(-aps)
    aps, names = aps[order], [names[i] for i in order]
    if top_k is not None:
        aps, names = aps[:top_k], names[:top_k]

    plt.figure(figsize=(max(6, 0.35*len(aps)), 4))
    plt.bar(range(len(aps)), aps)
    plt.xticks(range(len(aps)), names, rotation=90)
    plt.ylim(0, 1.0)
    plt.ylabel("AP"); plt.title("Average Precision par classe")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); plt.close()
    else: plt.show()


# =========================
# 4) Matrices de confusion par classe (@0.5)
# =========================
def plot_confusion_matrices(probs, targets, class_names=None, save_path=None, threshold=0.5, limit_classes=None):
    bin_preds = (probs >= threshold).astype(int)
    conf_matrices = multilabel_confusion_matrix(targets, bin_preds)
    n_classes = targets.shape[1]
    idx_range = range(n_classes) if limit_classes is None else range(min(limit_classes, n_classes))
    for i in idx_range:
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_matrices[i], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        plt.title(f"Confusion — {class_names[i] if class_names else f'Class {i}'}")
        plt.ylabel("Vrai"); plt.xlabel("Prédit")
        plt.tight_layout()
        if save_path: plt.savefig(f"{save_path}_class_{i}.png", dpi=300); plt.close()
        else: plt.show()


# =========================
# 5) Matrice de confusion globale (tous labels à plat)
# =========================
def plot_global_confusion_matrix(probs, targets, threshold=0.5, save_path=None):
    bin_preds = (probs >= threshold).astype(int)
    y_true_flat = targets.ravel()
    y_pred_flat = bin_preds.ravel()
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Matrice de confusion globale (@0.5)")
    plt.ylabel("Vrai"); plt.xlabel("Prédit")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); plt.close()
    else: plt.show()


# =========================
# 6) Courbes d’apprentissage adaptées à ton history
# =========================
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(history, save_path=None):
    # Accepte (history, ...) ou history seul
    if isinstance(history, tuple):
        history = history[0]
    assert isinstance(history, dict), "history doit être un dict."

    # Longueur des époques (prend la première clé présente)
    for k in ["train_loss","val_loss","val_f1","train_f1","val_mAP"]:
        if k in history and isinstance(history[k], (list, tuple)):
            T = len(history[k]); break
    else:
        raise ValueError("Impossible d'inférer le nombre d'époques depuis history.")
    epochs = np.arange(1, T+1)

    def get(key):
        v = history.get(key, None)
        if isinstance(v, (list, tuple)): v = v[:T]
        return v

    train_loss = get("train_loss")
    val_loss   = get("val_loss")

    train_f1   = get("train_f1")
    val_f1     = get("val_f1")
    # Tu stockes *opt_stable*
    val_f1_opt = get("val_f1_opt_stable")

    train_EMA  = get("train_EMA")
    val_EMA    = get("val_EMA")
    val_EMA_opt= get("val_EMA_opt_stable")

    val_mAP    = get("val_mAP") or get("val_mAP_micro")
    val_HAM    = get("val_HAM")
    val_HAM_opt= get("val_HAM_opt_stable")

    plt.figure(figsize=(14,8))

    # (a) Loss
    plt.subplot(2,2,1)
    if train_loss is not None: plt.plot(epochs, train_loss, label="Train")
    if val_loss   is not None: plt.plot(epochs, val_loss,   label="Val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(alpha=0.3)

    # (b) F1 micro
    plt.subplot(2,2,2)
    if train_f1 is not None: plt.plot(epochs, train_f1, label="Train@0.5")
    if val_f1   is not None: plt.plot(epochs, val_f1,   label="Val@0.5")
    if val_f1_opt is not None: plt.plot(epochs, val_f1_opt, "--", label="Val@opt(stable)")
    plt.title("F1 (micro)"); plt.xlabel("Epoch"); plt.ylabel("F1")
    plt.legend(); plt.grid(alpha=0.3)

    # (c) EMA
    plt.subplot(2,2,3)
    if train_EMA is not None: plt.plot(epochs, train_EMA, label="Train@0.5")
    if val_EMA   is not None: plt.plot(epochs, val_EMA,   label="Val@0.5")
    if val_EMA_opt is not None: plt.plot(epochs, val_EMA_opt, "--", label="Val@opt(stable)")
    plt.title("Exact Match Accuracy"); plt.xlabel("Epoch"); plt.ylabel("EMA")
    plt.legend(); plt.grid(alpha=0.3)

    # (d) mAP & Hamming
    plt.subplot(2,2,4)
    if val_mAP is not None: plt.plot(epochs, val_mAP, label="Val mAP (micro)")
    if val_HAM is not None: plt.plot(epochs, val_HAM, label="Val HAM@0.5")
    if val_HAM_opt is not None: plt.plot(epochs, val_HAM_opt, "--", label="Val HAM@opt(stable)")
    plt.title("mAP & Hamming"); plt.xlabel("Epoch")
    plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); plt.close()
    else: plt.show()
