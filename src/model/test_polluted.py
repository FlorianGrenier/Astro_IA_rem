import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

one_polluted = sorted(glob.glob("data_generate/synthetic_spectra_polluted_batch_*.npy"))[0]
one_labels   = sorted(glob.glob("data_generate/synthetic_labels_corrected_batch_*.npy"))[0]

Xp = np.load(one_polluted)   # (B, Nch)
Yc = np.load(one_labels)     # (B, C)


# Choisis l'index du spectre à inspecter
idx = 3  # modifie si besoin

B, Nch = Xp.shape
print(f"Batch size={B}, N_channels={Nch}, nb_classes={Yc.shape[1]}")
print(f"Index choisi: {idx}")

spec = Xp[idx]          # intensités (Nch,)
labels = Yc[idx]        # labels binaires (C,)

# Stats rapides
print("\n--- Stats du spectre ---")
print(f"min={spec.min():.6f}  max={spec.max():.6f}  mean={spec.mean():.6f}  std={spec.std():.6f}")

# Aperçu des premières valeurs (sans tout spammer la console)
print("\n--- Aperçu des 20 premières intensités ---")
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print(spec[:20])

# Aperçu des labels
print("\n--- Labels (indices des molécules présentes) ---")
present_idx = np.where(labels == 1)[0]
print(present_idx.tolist())

print(labels)