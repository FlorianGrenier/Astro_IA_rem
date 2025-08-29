import os
import numpy as np
import torch
from torch.utils.data import Dataset
from bisect import bisect_right

class SpectralBatchDataset(Dataset):
    """
    Dataset paresseux pour lots .npy (X: spectres, Y: labels).
    - X_fichier: (B_i, NCH) float
    - Y_fichier: (B_i, C)   {0,1}
    Retour: x: torch.float32 (1, NCH), y: torch.float32 (C,)
    """
    def __init__(
        self,
        spectra_files,
        labels_files,
        transform=None,          # callable sur x (numpy (1,NCH) ou torch (1,NCH))
        return_index: bool = False,
        dtype_x=np.float32,
        dtype_y=np.float32,
    ):
        assert len(spectra_files) == len(labels_files), "Mismatch nb de fichiers X/Y"
        self.sfiles = list(spectra_files)
        self.lfiles = list(labels_files)
        self.transform = transform
        self.return_index = return_index
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y

        # Ouverture memmap (lecture seule)
        self._Xs, self._Ys = [], []
        self._lengths = []
        self.NCH = None
        self.C = None

        for xp, yp in zip(self.sfiles, self.lfiles):
            assert os.path.exists(xp) and os.path.exists(yp), f"Fichier manquant: {xp} / {yp}"
            X = np.load(xp, mmap_mode='r')  # (Bi, NCH)
            Y = np.load(yp, mmap_mode='r')  # (Bi, C)
            assert len(X) == len(Y), f"Tailles différentes pour {xp} et {yp}"

            if self.NCH is None:
                self.NCH = int(X.shape[1])
            else:
                assert X.shape[1] == self.NCH, "NCH incohérent entre fichiers X"
            if self.C is None:
                self.C = int(Y.shape[1])
            else:
                assert Y.shape[1] == self.C, "Nb de classes incohérent entre fichiers Y"

            self._Xs.append(X)
            self._Ys.append(Y)
            self._lengths.append(len(X))

        self._cumlen = np.cumsum(self._lengths)
        self._N = int(self._cumlen[-1]) if len(self._cumlen) > 0 else 0
        assert self._N > 0, "Dataset vide."

    def __len__(self):
        return self._N

    def _locate(self, idx: int):
        i = bisect_right(self._cumlen, idx)
        prev = 0 if i == 0 else self._cumlen[i - 1]
        off = idx - prev
        return i, off

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._N:
            raise IndexError(idx)

        i, off = self._locate(idx)

        
        x_np = self._Xs[i][off].astype(self.dtype_x, copy=False)   # (NCH,)
        y_np = self._Ys[i][off].astype(self.dtype_y, copy=False)   # (C,)

        x_np = np.expand_dims(x_np, axis=0)                        # (1, NCH)

        
        x_np = np.ascontiguousarray(x_np).copy()
        y_np = np.ascontiguousarray(y_np).copy()


        if self.transform is not None:
            out = self.transform(x_np)
            # supporte transform qui retourne numpy OU torch
            if isinstance(out, np.ndarray):
                x_np = out
            elif torch.is_tensor(out):
                x = out.float()
            else:
                raise TypeError("transform doit retourner np.ndarray ou torch.Tensor")
        # Conversion torch si pas déjà fait
        if 'x' not in locals():
            x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        if self.return_index:
            return x, y, idx
        return x, y


