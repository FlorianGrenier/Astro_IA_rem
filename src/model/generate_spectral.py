import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from pretraitement import generate_multiple_spectra, generate_corrected_labels
from tqdm import tqdm
import gc

np.random.seed(42)

folder = "/home/flopops/Documents/Astro_IA/src/data/nouvelle_molecule/MODELS"


files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))

spectra_data = []
molecule_names = []

for file in files:
    name = os.path.splitext(file)[0] 
    path = os.path.join(folder, file)
    try:
        data = np.loadtxt(path, skiprows=1)
        freqs, intens = data[:,0], data[:,1]

        spectra_data.append((freqs, intens))
        
        molecule_names.append(name)
        print(f"Loaded {name}, length = {len(freqs)}")
    
    except Exception as e:
        print(f"  Error in {file}: {e}")

f_min = max(np.min(f) for f, _ in spectra_data)
f_max = min(np.max(f) for f, _ in spectra_data)
Nch   = min(len(f) for f, _ in spectra_data) 
grid_freqs = np.linspace(f_min, f_max, Nch)

def resample_to_grid(freqs, intens, grid):
   
    return np.interp(grid, freqs, intens, left=0.0, right=0.0)

spectra_data = [(grid_freqs, resample_to_grid(f, x, grid_freqs)) for (f, x) in spectra_data]


os.makedirs("data_generate", exist_ok=True)


np.save("data_generate/frequencies.npy", grid_freqs)

total_spectra = 50000
batch_size = 2500

all_counts = None
names_list = None

for i in tqdm(range(0, total_spectra, batch_size), desc="Batches"):
    batch_idx = i // batch_size + 1
    print(f"Processing batch {batch_idx}...")
    dataset = generate_multiple_spectra(
        spectra_data, molecule_names, num_spectra=batch_size, max_molecules=8
    )
    
    
    sigmas = dataset.get("sigma", None)
    if sigmas is not None:
        sigmas = np.asarray(sigmas, dtype=float)

    corrected_labels = generate_corrected_labels(
        dataset["pollution"],          
        dataset["labels"],             
        dataset["base_names"],          
        spectra_data,                    
        molecule_names,
        std_noise=sigmas                 
    )

    X_syn = dataset["clean"].astype(np.float32)
    X_noise = dataset["noise"].astype(np.float32)
    X_polluted = dataset["pollution"].astype(np.float32)
    Y_syn = dataset["labels"].astype(np.int8)
    Y_corrected = corrected_labels.astype(np.int8)

    np.save(f"data_generate/synthetic_spectra_batch_{batch_idx}.npy", X_syn)
    np.save(f"data_generate/synthetic_spectra_noise_batch_{batch_idx}.npy", X_noise)
    np.save(f"data_generate/synthetic_spectra_polluted_batch_{batch_idx}.npy", X_polluted)
    np.save(f"data_generate/synthetic_labels_batch_{batch_idx}.npy", Y_syn)
    np.save(f"data_generate/synthetic_labels_corrected_batch_{batch_idx}.npy", Y_corrected)


    if names_list is None:
        names_list = dataset["base_names"]
    
    batch_counts = np.sum(Y_syn, axis=0)
    all_counts = batch_counts if all_counts is None else (all_counts + batch_counts)

    print(f"Batch {i // batch_size + 1} processed. Total spectra: {len(X_syn)}")
    print(f"Counts for this batch: {batch_counts}")
    
    del X_syn, Y_syn, dataset
    gc.collect()


for name, cnt in zip(names_list, all_counts):
    print(f"{name}: {cnt} occurrences")

with open("molecule_names.txt", "w") as f:
    for name in names_list:
        f.write(name + "\n")
print("Synthetic spectra and labels saved successfully.")