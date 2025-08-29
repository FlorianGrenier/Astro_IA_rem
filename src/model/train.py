# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, hamming_loss, average_precision_score,
    precision_score, recall_score, precision_recall_curve
)

# -------------------- Métriques --------------------
def exact_match_accuracy_bin(y_pred_bin: np.ndarray, y_true_bin: np.ndarray) -> float:
    return float((y_pred_bin == y_true_bin).all(axis=1).mean())

def macro_map(probs: np.ndarray, y_true: np.ndarray) -> float:
    C = y_true.shape[1]
    aps = []
    for i in range(C):
        yi = y_true[:, i]
        pi = probs[:, i]
        if yi.max() == yi.min():
            continue
        aps.append(average_precision_score(yi, pi))
    return float(np.mean(aps)) if aps else 0.0

# -------------------- Seuils --------------------
def optimal_thresholds_per_class(probs: np.ndarray, y_true: np.ndarray, max_grid: int = 200) -> np.ndarray:
    """Seuils par QUANTILES (F1 max par classe sur VAL)."""
    C = y_true.shape[1]
    thrs = np.full(C, 0.5, dtype=np.float32)
    for i in range(C):
        p = probs[:, i].astype(np.float64)
        t = y_true[:, i].astype(np.int32)
        if t.max() == t.min():
            continue
        qs = np.linspace(0.0, 1.0, num=min(max_grid, len(p) + 2), endpoint=True)
        cand = np.unique(np.quantile(p, qs))
        cand = np.clip(cand, 1e-6, 1 - 1e-6)
        best, best_thr = -1.0, 0.5
        for thr in cand:
            pred = (p >= thr).astype(np.int32)
            score = f1_score(t, pred, zero_division=0)
            if score > best:
                best, best_thr = score, thr
        thrs[i] = np.float32(best_thr)
    return thrs

def optimal_thresholds_per_class_pr(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Seuils via courbe P–R (F1 max)."""
    C = y_true.shape[1]
    thrs = np.full(C, 0.5, dtype=np.float32)
    for i in range(C):
        p = probs[:, i].astype(np.float64)
        t = y_true[:, i].astype(np.int32)
        if t.max() == t.min():
            continue
        prec, rec, thr_raw = precision_recall_curve(t, p)
        if len(thr_raw) == 0:
            continue
        denom = (prec[1:] + rec[1:]).clip(min=1e-12)
        f1 = 2 * prec[1:] * rec[1:] / denom
        j = int(np.argmax(f1))
        thrs[i] = np.float32(np.clip(thr_raw[j], 1e-6, 1 - 1e-6))
    return thrs

def apply_thresholds(probs: np.ndarray, thr_per_class: np.ndarray | float = 0.5) -> np.ndarray:
    if np.isscalar(thr_per_class):
        return (probs >= float(thr_per_class)).astype(np.int32)
    thr = np.asarray(thr_per_class, dtype=np.float32)[None, :]
    return (probs >= thr).astype(np.int32)

def metrics_from_probs(probs: np.ndarray, y_true: np.ndarray, thr_or_thrs: np.ndarray | float):
    y_pred = apply_thresholds(probs, thr_or_thrs)
    f1_micro = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    precision = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    recall    = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    ema = exact_match_accuracy_bin(y_pred, y_true)
    ham = hamming_loss(y_true, y_pred)
    return f1_micro, ema, ham, precision, recall

# -------------------- Boucles train/eval --------------------
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 1.0) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for xb, yb in tqdm(loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = xb.size(0)
        loss_sum += loss.item() * bs
        n += bs
        del logits, loss, xb, yb
    return loss_sum / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device, thr: float = 0.5, criterion=None):
    model.eval()
    all_logits, all_targets = [], []
    running_loss, n = 0.0, 0
    for xb, yb in tqdm(loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        if criterion is not None:
            loss = criterion(logits, yb)
            bs = xb.size(0)
            running_loss += loss.item() * bs
            n += bs
        all_logits.append(logits.detach().cpu())
        all_targets.append(yb.detach().cpu())
        del logits, xb, yb

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.sigmoid(logits).numpy()
    y_true = targets.numpy()

    # @0.5
    y_pred = (probs >= thr).astype(np.int32)
    f1_micro = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)

    y_flat = y_true.flatten()
    p_flat = probs.flatten()
    ap_micro = float('nan') if y_flat.max() == y_flat.min() else average_precision_score(y_flat, p_flat)

    ema = exact_match_accuracy_bin(y_pred, y_true)
    ham = hamming_loss(y_true, y_pred)
    val_loss = (running_loss / max(n, 1)) if criterion is not None else None
    return f1_micro, ap_micro, ema, ham, probs, y_true, val_loss

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs: int = 30,
    scheduler=None,
    save_path: str = "best_model_simple.pth",
    patience: int = 6,
    log_train_metrics: bool = True,
    thr_method: str = "quantiles",   # "quantiles" ou "pr"
    recompute_thr_every: int = 5,    # recalcul périodique
    freeze_thr_after: int | None = 10,  # fige après cette époque (None = jamais)
):
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "train_mAP": [], "train_EMA": [], "train_HAM": [],
        "val_f1": [], "val_mAP": [], "val_EMA": [], "val_HAM": [],
        # métriques "stables" (avec meilleurs seuils connus)
        "val_f1_opt_stable": [], "val_EMA_opt_stable": [], "val_HAM_opt_stable": [],
    }
    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    thr_save_path = os.path.splitext(save_path)[0] + "_thr.npy"

    thr_func = optimal_thresholds_per_class if thr_method == "quantiles" else optimal_thresholds_per_class_pr
    last_thr_per_class = None       # derniers seuils calculés
    best_thr_per_class = None       # meilleurs seuils (ceux qui donnent la meilleure val_f1_opt)
    thresholds_frozen = False

    for ep in tqdm(range(1, epochs + 1), desc="Epochs"):
        # --- Train ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # --- Eval @0.5 ---
        if log_train_metrics:
            f1_tr, ap_tr, ema_tr, ham_tr, _, _, _ = evaluate(
                model, train_loader, device, thr=0.5, criterion=criterion
            )
        else:
            f1_tr = ap_tr = ema_tr = ham_tr = np.nan

        f1_v, ap_v, ema_v, ham_v, probs_v, y_v, val_loss = evaluate(
            model, val_loader, device, thr=0.5, criterion=criterion
        )

        # --- Recompute périodique des seuils (si pas figés) ---
        if (not thresholds_frozen) and (recompute_thr_every > 0) and (ep % recompute_thr_every == 0):
            last_thr_per_class = thr_func(probs_v, y_v).astype(np.float32)

            # évalue la qualité avec ces seuils et garde le meilleur "opt_stable"
            val_f1_opt, val_ema_opt, val_ham_opt, _, _ = metrics_from_probs(probs_v, y_v, last_thr_per_class)
            if (best_thr_per_class is None) or (val_f1_opt > metrics_from_probs(probs_v, y_v, best_thr_per_class)[0]):
                best_thr_per_class = last_thr_per_class.copy()

        # --- Freeze des seuils après une certaine époque ---
        if (freeze_thr_after is not None) and (ep >= freeze_thr_after) and (not thresholds_frozen):
            # si on n'a pas encore de "best", calcule-le maintenant
            if best_thr_per_class is None:
                if last_thr_per_class is None:
                    last_thr_per_class = thr_func(probs_v, y_v).astype(np.float32)
                best_thr_per_class = last_thr_per_class.copy()
            thresholds_frozen = True

        # --- Log des métriques "stables" (avec meilleurs seuils connus) ---
        if best_thr_per_class is not None:
            val_f1_opt_stable, val_ema_opt_stable, val_ham_opt_stable, _, _ = metrics_from_probs(
                probs_v, y_v, best_thr_per_class
            )
        else:
            val_f1_opt_stable = val_ema_opt_stable = val_ham_opt_stable = np.nan

        # --- Logs ---
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss if val_loss is not None else np.nan)
        history["train_f1"].append(f1_tr); history["train_mAP"].append(ap_tr); history["train_EMA"].append(ema_tr); history["train_HAM"].append(ham_tr)
        history["val_f1"].append(f1_v);   history["val_mAP"].append(ap_v);    history["val_EMA"].append(ema_v);    history["val_HAM"].append(ham_v)
        history["val_f1_opt_stable"].append(val_f1_opt_stable)
        history["val_EMA_opt_stable"].append(val_ema_opt_stable)
        history["val_HAM_opt_stable"].append(val_ham_opt_stable)

        # --- Scheduler & print ---
        if scheduler is not None:
            try: scheduler.step(f1_v)
            except TypeError: scheduler.step()

        msg = (
            f"[Epoch {ep:02d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1@0.5={f1_v:.3f} | val_mAP={ap_v:.3f} | val_EMA@0.5={ema_v:.3f} | val_HAM@0.5={ham_v:.4f}"
            f" || train_f1@0.5={f1_tr:.3f} | train_mAP={ap_tr:.3f} | train_EMA@0.5={ema_tr:.3f} | train_HAM@0.5={ham_tr:.4f}"
        )
        if not np.isnan(val_f1_opt_stable):
            msg += f" || val_f1@opt(stable)={val_f1_opt_stable:.3f} | val_EMA@opt(stable)={val_ema_opt_stable:.3f}"
            msg += f" | val_HAM@opt(stable)={val_ham_opt_stable:.4f}"
        print(msg)

        # --- Early stopping / save sur f1@0.5 ---
        if f1_v > best_val_f1 + 1e-4:
            best_val_f1 = f1_v
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            # si les seuils sont figés, sauvegarde-les aussi
            if thresholds_frozen and (best_thr_per_class is not None):
                np.save(thr_save_path, best_thr_per_class.astype(np.float32))
            print(f"  ↳ new best (val_f1@0.5={best_val_f1:.3f}), saved to {save_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no improvement {no_improve} ≥ {patience})")
                break

    # ---- Fin d’entraînement : charge best weights et calcule/sauvegarde seuils propres ----
    if best_state is not None:
        model.load_state_dict(best_state)
    # calcule une fois des seuils propres sur VAL si on n’a rien figé
    if best_thr_per_class is None:
        _, _, _, _, probs_v, y_v, _ = evaluate(model, val_loader, device, thr=0.5, criterion=None)
        best_thr_per_class = thr_func(probs_v, y_v).astype(np.float32)

    np.save(thr_save_path, best_thr_per_class.astype(np.float32))
    print(f"[SAVE] Thresholds saved to {thr_save_path}")

    return history, best_thr_per_class


# -------------------- Test --------------------
@torch.no_grad()
def test_model(
    model,
    test_loader,
    device,
    path: str | None = None,
    thr_per_class: np.ndarray | None = None,
    allow_learn_thresholds_on_test: bool = False,
    thr_method: str = "quantiles",
):
    if path and os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        print(f"[LOAD] Best weights loaded from {path}")
        if thr_per_class is None:
            thr_path = os.path.splitext(path)[0] + "_thr.npy"
            if os.path.exists(thr_path):
                thr_per_class = np.load(thr_path).astype(np.float32)
                print(f"[LOAD] Thresholds loaded from {thr_path}")

    # 1) test @0.5
    f1_t, ap_micro_t, ema_t, ham_t, probs_t, yt, _ = evaluate(model, test_loader, device, thr=0.5, criterion=None)
    map_macro_t = macro_map(probs_t, yt)
    print(f"[TEST @0.5]  F1_micro={f1_t:.3f} | mAP_micro={ap_micro_t:.3f} | mAP_macro={map_macro_t:.3f} | EMA={ema_t:.3f} | HAM={ham_t:.4f}")

    # 2) test @opt
    if thr_per_class is None:
        if allow_learn_thresholds_on_test:
            thr_func = optimal_thresholds_per_class if thr_method == "quantiles" else optimal_thresholds_per_class_pr
            thr_per_class = thr_func(probs_t, yt).astype(np.float32)
            print("[WARN] thr_per_class non fourni: calculé sur TEST (préférez VAL).")
        else:
            print("[INFO] Aucun `thr_per_class` fourni. Skipping TEST @opt (passez les seuils appris sur la VAL).")
            return probs_t, yt, None

    pred_opt = apply_thresholds(probs_t, thr_per_class)
    f1_micro_opt = f1_score(yt.flatten(), pred_opt.flatten(), zero_division=0)
    ema_opt = exact_match_accuracy_bin(pred_opt, yt)
    ham_opt = hamming_loss(yt, pred_opt)
    print(f"[TEST @opt]  F1_micro={f1_micro_opt:.3f} | EMA={ema_opt:.3f} | HAM={ham_opt:.4f}")

    return probs_t, yt, thr_per_class
