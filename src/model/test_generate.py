import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from collections import Counter

# --- 0) Chargement ---
freqs = np.load("data_generate/frequencies.npy")
batch_num = 20

X_syn      = np.load(f"data_generate/synthetic_spectra_batch_{batch_num}.npy")
X_noise    = np.load(f"data_generate/synthetic_spectra_noise_batch_{batch_num}.npy")
X_polluted = np.load(f"data_generate/synthetic_spectra_polluted_batch_{batch_num}.npy")
Y_syn      = np.load(f"data_generate/synthetic_labels_batch_{batch_num}.npy")
Y_corrected= np.load(f"data_generate/synthetic_labels_corrected_batch_{batch_num}.npy")

# essaie d'abord dans data_generate/, sinon à la racine
names_path = "data_generate/molecule_names.txt"
if not os.path.exists(names_path):
    names_path = "molecule_names.txt"

with open(names_path, "r") as f:
    molecule_names = [line.strip() for line in f if line.strip()]

def get_molecule_names(label_vector):
    idxs = np.where(label_vector)[0]
    return [molecule_names[i] for i in idxs]

# --- 1) Différences de labels ---
diff_indices = np.where(np.any(Y_syn != Y_corrected, axis=1))[0]
print(f"Spectres avec correction de labels (batch {batch_num}) : {diff_indices.tolist()}")

if len(diff_indices) == 0:
    print("Aucune différence de labels sur ce batch. Rien à tracer.")
    raise SystemExit

# --- 2) Résumé global (compteurs + CSV) ---
added_counter   = Counter()
removed_counter = Counter()

summary_rows = []
for idx in diff_indices:
    before = set(get_molecule_names(Y_syn[idx]))
    after  = set(get_molecule_names(Y_corrected[idx]))
    added   = sorted(after - before)
    removed = sorted(before - after)
    for m in added:   added_counter[m]   += 1
    for m in removed: removed_counter[m] += 1
    summary_rows.append({
        "index": idx,
        "before": ";".join(sorted(before)),
        "after":  ";".join(sorted(after)),
        "added":  ";".join(added) if added else "",
        "removed":";".join(removed) if removed else ""
    })

os.makedirs("summaries", exist_ok=True)
import csv
csv_path = f"summaries/diff_summary_batch_{batch_num}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["index","before","after","added","removed"])
    w.writeheader()
    for row in summary_rows:
        w.writerow(row)
print(f"Résumé CSV écrit : {csv_path}")

print("\nTop molécules AJOUTÉES (après correction) :")
for m, c in added_counter.most_common(10):
    print(f"  {m}: +{c}")

print("\nTop molécules SUPPRIMÉES (après correction) :")
for m, c in removed_counter.most_common(10):
    print(f"  {m}: -{c}")

# --- 3) Fonctions utilitaires pour le zoom auto ---
def auto_zoom_limits(freqs, y, window_pts=500):
    """centre une fenêtre autour du pic le plus fort."""
    if len(y) == 0:
        return (freqs[0], freqs[-1])
    j = int(np.argmax(y))
    j0 = max(0, j - window_pts//2)
    j1 = min(len(y)-1, j + window_pts//2)
    return float(freqs[j0]), float(freqs[j1])

# --- 4) Visualisation (tu peux limiter à N exemples si c’est trop) ---
output_dir = f"spectra_corrected_batch_{batch_num}"
os.makedirs(output_dir, exist_ok=True)

# Option: limite à 50 figures si tu as trop de différences
MAX_FIGS = None  # ex. 50
count = 0

for idx in diff_indices:
    before = set(get_molecule_names(Y_syn[idx]))
    after  = set(get_molecule_names(Y_corrected[idx]))
    if before == after:
        continue

    added   = sorted(after - before)
    removed = sorted(before - after)
    mols_before = sorted(before)
    mols_after  = sorted(after)

    print(f"\nSpectre {idx} :")
    print(" - Avant :", mols_before)
    print(" - Après :", mols_after)
    if len(added) or len(removed):
        print(" - Ajoutées :", added if added else "Aucune")
        print(" - Supprimées :", removed if removed else "Aucune")

    # --- figure complète ---
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(freqs, X_syn[idx], linewidth=1)
    ax1.set_title(f"Spectre {idx} - Propre")
    ax1.set_ylabel("Intensité")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(freqs, X_noise[idx], alpha=0.8, linewidth=0.8)
    ax2.set_title("Spectre Bruité")
    ax2.set_ylabel("Intensité")
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(freqs, X_polluted[idx], alpha=0.8, linewidth=0.8)
    ax3.set_title("Spectre Pollué")
    ax3.set_xlabel("Fréquence (GHz)")  # <- corrige l’unité
    ax3.set_ylabel("Intensité")
    ax3.grid(True)

    fig.text(0.10, 0.80, "Molécules (propre):\n" + ("\n".join(mols_before) if mols_before else "Aucune"),
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    fig.text(0.30, 0.80, "Molécules (corrigées):\n" + ("\n".join(mols_after) if mols_after else "Aucune"),
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    fig.text(0.55, 0.94, "Ajoutées:\n" + ("\n".join(added) if added else "Aucune"),
             fontsize=10, bbox=dict(facecolor='lightgreen', alpha=0.8))
    fig.text(0.75, 0.94, "Supprimées:\n" + ("\n".join(removed) if removed else "Aucune"),
             fontsize=10, bbox=dict(facecolor='salmon', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, f"spectrum_comparison_corrected_{idx}.png"), dpi=150)
    plt.close()

    # --- figure zoom auto autour d’un pic fort du spectre pollué ---
    f0, f1 = auto_zoom_limits(freqs, X_polluted[idx], window_pts=800)

    figz = plt.figure(figsize=(15, 10))
    gsz = GridSpec(3, 1, figure=figz)

    ax1z = figz.add_subplot(gsz[0])
    ax1z.plot(freqs, X_syn[idx], linewidth=1)
    ax1z.set_xlim(f0, f1)
    ax1z.set_title(f"Spectre {idx} - Propre (Zoom)")
    ax1z.set_ylabel("Intensité")
    ax1z.grid(True)

    ax2z = figz.add_subplot(gsz[1])
    ax2z.plot(freqs, X_noise[idx], alpha=0.8, linewidth=0.8)
    ax2z.set_xlim(f0, f1)
    ax2z.set_title("Spectre Bruité (Zoom)")
    ax2z.set_ylabel("Intensité")
    ax2z.grid(True)

    ax3z = figz.add_subplot(gsz[2])
    ax3z.plot(freqs, X_polluted[idx], alpha=0.8, linewidth=0.8)
    ax3z.set_xlim(f0, f1)
    ax3z.set_title("Spectre Pollué (Zoom)")
    ax3z.set_xlabel("Fréquence (GHz)")
    ax3z.set_ylabel("Intensité")
    ax3z.grid(True)

    figz.text(0.10, 0.80, "Molécules (propre):\n" + ("\n".join(mols_before) if mols_before else "Aucune"),
              fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    figz.text(0.30, 0.80, "Molécules (corrigées):\n" + ("\n".join(mols_after) if mols_after else "Aucune"),
              fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    figz.text(0.55, 0.94, "Ajoutées:\n" + ("\n".join(added) if added else "Aucune"),
              fontsize=10, bbox=dict(facecolor='lightgreen', alpha=0.8))
    figz.text(0.75, 0.94, "Supprimées:\n" + ("\n".join(removed) if removed else "Aucune"),
              fontsize=10, bbox=dict(facecolor='salmon', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, f"spectrum_comparison_corrected_{idx}_zoom.png"), dpi=150)
    plt.close()

    count += 1
    if MAX_FIGS is not None and count >= MAX_FIGS:
        print(f"Limité à {MAX_FIGS} figures.")
        break
