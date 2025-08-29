# model_spectrum_baseline_v2.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- petits utilitaires ----------
def GN(ch: int, pref_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm robuste: adapte le nb de groupes à ch (évite les erreurs de divisibilité)."""
    g = math.gcd(ch, pref_groups) or 1
    return nn.GroupNorm(g, ch)

class ConvGNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=5, dilation=1):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, dilation=dilation, bias=False)
        self.gn   = GN(c_out, 8)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class SE1D(nn.Module):
    """Squeeze-and-Excitation léger pour 1D."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        hidden = max(1, ch // r)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Conv1d(ch, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class ResSE(nn.Module):
    """Petit bloc résiduel + SE. Proj 1x1 si changement de canaux."""
    def __init__(self, c_in, c_out, k=5, dilation=1, p_drop: float = 0.0):
        super().__init__()
        self.conv1 = ConvGNReLU(c_in,  c_out, k=k, dilation=dilation)
        # deuxième conv sans ReLU à la fin (on l'applique après l'addition)
        pad2 = (3 // 2) * dilation
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=pad2, dilation=dilation, bias=False),
            GN(c_out, 8)
        )
        self.se = SE1D(c_out, r=8)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.proj = None
        if c_in != c_out:
            self.proj = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
                GN(c_out, 8)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.se(out)
        out = out + identity
        out = self.act(out)
        out = self.drop(out)
        return out

# --------- Modèle CNN 1D “base saine ++ léger” ----------
class SpectrumCNN(nn.Module):
    """
    CNN 1D simple et efficace pour classification multi-label de spectres.
    - Entrée  : (B, in_ch, L)
    - Sortie  : (B, num_classes) (logits -> BCEWithLogitsLoss)
    Changements vs baseline:
      * Ajout d'un bloc résiduel + SE par stage (A, B, C, D)
      * Même schéma de décimation: AvgPool1d 4x à chaque stage
    """
    def __init__(self, num_classes: int, in_ch: int = 3, c: int = 64, p_drop: float = 0.3):
        super().__init__()

        # --- Stage A (L -> L/4) ---
        self.A = nn.Sequential(
            ResSE(in_ch, c,   k=5, dilation=1, p_drop=0.0),
            nn.AvgPool1d(kernel_size=4, stride=4)   # ↓ /4
        )

        # --- Stage B (L/4 -> L/16) ---
        self.B = nn.Sequential(
            ResSE(c, 2*c, k=5, dilation=1, p_drop=0.0),
            nn.AvgPool1d(kernel_size=4, stride=4)   # ↓ /16 total
        )

        # --- Stage C (L/16 -> L/64) ---
        self.C = nn.Sequential(
            ResSE(2*c, 2*c, k=5, dilation=1, p_drop=0.0),
            nn.AvgPool1d(kernel_size=4, stride=4)   # ↓ /64 total
        )

        # --- Bloc dilaté (pas de réduction) ---
        self.D = nn.Sequential(
            ResSE(2*c, 2*c, k=5, dilation=2, p_drop=p_drop),
        )

        # --- Tête compacte (inchangée pour garder la simplicité/paramètres) ---
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),   # (B, 2c)
            nn.Linear(2*c, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes)  # logits
        )

    def forward(self, x):
        x = self.A(x)   # (B, c,   L/4)
        x = self.B(x)   # (B, 2c,  L/16)
        x = self.C(x)   # (B, 2c,  L/64)
        x = self.D(x)   # (B, 2c,  L/64)
        x = self.head(x)
        return x
