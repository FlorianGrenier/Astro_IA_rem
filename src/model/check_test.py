# === 3) QC : charger un batch et visualiser quelques exemples ===
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

one_polluted = sorted(glob.glob("data_generate/synthetic_spectra_polluted_batch_*.npy"))[0]
one_labels   = sorted(glob.glob("data_generate/synthetic_labels_corrected_batch_*.npy"))[0]

Xp = np.load(one_polluted)   # (B, Nch)
Yc = np.load(one_labels)     # (B, C)
freqs = np.load("data_generate/frequencies.npy")
with open("molecule_names.txt") as f:
    base_names = [l.strip() for l in f]

print("Exemple batch shapes:", Xp.shape, Yc.shape, "Mol classes:", len(base_names))
print(Xp.min(), Xp.max(), Yc.min(), Yc.max())
# --- Plot 3 spectres ---
plt.figure(figsize=(10,6))
for k in range(3):
    plt.plot(freqs, Xp[k], alpha=0.8, label=f"spec {k} | {Yc[k].sum()} mol")
plt.xlabel("Frequency")
plt.ylabel("Intensity (arb.)")
plt.legend()
plt.title("Exemples de spectres (polluted)")
plt.tight_layout()
plt.savefig("qc_spectres_exemples.png", dpi=300)   # <--- ENREGISTRE
plt.close()

# --- Distribution du nb de molécules par spectre ---
vals = Yc.sum(axis=1)
plt.figure()
plt.hist(vals, bins=np.arange(vals.min(), vals.max()+2)-0.5)
plt.xlabel("# molécules présentes")
plt.ylabel("Compte")
plt.title("Cardinalité des labels (QC)")
plt.tight_layout()
plt.savefig("qc_cardinalite_labels.png", dpi=300)  # <--- ENREGISTRE
plt.close()
