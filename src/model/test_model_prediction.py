import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import pandas as pd
import pickle


from model import SpectralCNN

print("📂 Chargement des fichiers...")

X_syn = np.load("data_generate/synthetic_spectra_test_noise_batch_1.npy")
Y_syn = np.load("data_generate/synthetic_labels_test_batch_1.npy")


with open("molecule_names.txt") as f:
    molecule_names_loaded = [line.strip() for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model = SpectralCNN(num_classes=Y_syn.shape[1])


model.load_state_dict(torch.load("best_model_noise.pth", map_location=device))
model.to(device)
model.eval()

# ====== Prédiction ======

def predict_spectrum(spectrum: np.ndarray, model: nn.Module, device: torch.device,
                     scaler=None, threshold: float = 0.5, return_probs=False):

    model.eval()

    if scaler is not None:
        spectrum = scaler.transform(spectrum.reshape(1, -1)).flatten()

    x = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).to(device) 

    with torch.no_grad():
        logits = model(x)  
        probs = torch.sigmoid(logits).cpu().numpy().flatten() 
        pred_labels = (probs >= threshold).astype(int)

    return (pred_labels, probs) if return_probs else pred_labels

# ====== Visualisation ======

def plot_prediction(molecule_names, probs, threshold=0.5):
    plt.figure(figsize=(10, 5))
    plt.bar(molecule_names, probs, color=["green" if p >= threshold else "gray" for p in probs])
    plt.axhline(y=threshold, color='red', linestyle='--', label=f"Seuil = {threshold}")
    plt.title("Probabilité par molécule détectée dans le spectre")
    plt.ylabel("Probabilité")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()




# ======  Affichage prédiction vs vérité terrain ======

def display_prediction_vs_ground_truth(spectrum_idx, X_data, Y_data, model, molecule_names, device, threshold=0.5):
    spectrum = X_data[spectrum_idx]
    true_labels = Y_data[spectrum_idx]

    pred_labels, probs = predict_spectrum(spectrum, model, device, threshold=threshold, return_probs=True)

    print(f"\n Spectre #{spectrum_idx}")
    print(f"{'Molécule':<20} | {'Probabilité':<10} | {'Prédit':<6} | {'Réel'}")
    print("-" * 55)

    for name, prob, pred, true in zip(molecule_names, probs, pred_labels, true_labels):
        status = "✅" if pred == true else "❌"
        print(f"{name:<20} | {prob:<10.3f} | {pred:<6} | {true} {status}")

    plot_prediction(molecule_names, probs, threshold=threshold)

# Exemple :
display_prediction_vs_ground_truth(
    spectrum_idx=3,
    X_data=X_syn,
    Y_data=Y_syn,
    model=model,
    molecule_names=molecule_names_loaded,
    device=device,
    threshold=0.5
)

# ====== 💾 Sauvegarde CSV ======

def save_predictions_to_csv(X_data, Y_data, model, molecule_names, device,
                            path="predictions_noise_with_ground_truth.csv", threshold=0.5, limit=100):

    results = []

    for i in range(min(limit, len(X_data))):
        spectrum = X_data[i]
        true_labels = Y_data[i]

        pred_labels, probs = predict_spectrum(spectrum, model, device, threshold=threshold, return_probs=True)
        row = {'index': i}

        for name, pred, prob, true in zip(molecule_names, pred_labels, probs, true_labels):
            row[f"{name}_pred"] = pred
            row[f"{name}_prob"] = round(prob, 4)
            row[f"{name}_true"] = true

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"\n📁 Fichier CSV (avec vérité terrain) sauvegardé sous : {path}")

# Sauvegarde les 100 premières prédictions
save_predictions_to_csv(X_syn, Y_syn, model, molecule_names_loaded, device)

from sklearn.metrics import classification_report

Y_true = Y_syn[:100]
Y_pred = []

for i in range(100):
    pred_labels = predict_spectrum(X_syn[i], model, device, threshold=0.5)
    Y_pred.append(pred_labels)

Y_pred = np.array(Y_pred)
print("\n📊 Rapport global (100 spectres) :")
print(classification_report(Y_true, Y_pred, target_names=molecule_names_loaded, zero_division=0))
