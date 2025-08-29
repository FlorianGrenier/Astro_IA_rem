import numpy as np
import matplotlib.pyplot as plt
import os
import torch


import numpy as np

from tqdm import tqdm




def normalize_minmax(s):
    s = np.asarray(s, dtype=float)
    vmin, vmax = float(np.min(s)), float(np.max(s))
    if vmax - vmin <= 1e-12:
        return np.zeros_like(s)
    return (s - vmin) / (vmax - vmin)


def get_base_molecule_name(full_name):
    parts = full_name.split('_')
    return parts[1] if len(parts) > 2 else full_name



def shift_spectrum(x, shift):
    x = np.asarray(x)
    if shift == 0:
        return x
    y = np.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]
    return y

def moving_avg(x, k=3):
    if k <= 1:
        return x
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    csum = np.cumsum(xpad, dtype=float)
    return (csum[k:] - csum[:-k]) / k

def add_gaussian_noise(spectrum):
    sigma = np.random.uniform(1e-3, 1e-1)
    noise = np.random.normal(loc=0.0, scale=sigma, size=len(spectrum))
    return spectrum + noise, float(sigma)


def add_pollution(spectrum, frac=0.02, amp_sigma=0.6):
    polluted = spectrum.copy()
    Nch = len(spectrum)
    n_lines = max(1, int(frac * Nch))
    scale = np.std(spectrum) if np.std(spectrum) > 0 else 0.05
    for _ in range(n_lines):
        center = np.random.randint(0, Nch)
        width = np.random.randint(1, 5)           # 1..4 canaux
        amp = np.random.lognormal(mean=0.0, sigma=amp_sigma) * scale
        for i in range(-width, width + 1):
            pos = center + i
            if 0 <= pos < Nch:
                polluted[pos] += amp * np.exp(-0.5 * (i / max(1, width / 2)) ** 2)
    return polluted

def compute_local_width(signal, peak_index, threshold):

    left = peak_index

    while left > 0 and signal[left] > threshold:
        left -= 1

    right = peak_index
    
    while right < len(signal) - 1 and signal[right] > threshold:
        right += 1

    return right - left



def merge_peaks_locally(candidates, signal, threshold):
    if not candidates:
        return []

    merged = []
    i = 0

    while i < len(candidates):
        current = candidates[i]
        current_width = compute_local_width(signal, current, threshold)
        j = i + 1

        MERGE_MARGIN = 5  
        while j < len(candidates) and candidates[j] - current < (current_width + MERGE_MARGIN):
            if signal[candidates[j]] > signal[current]:
                current = candidates[j]
                current_width = compute_local_width(signal, current, threshold)
            j += 1

        merged.append(current)
        i = j

    return merged





def detect_peaks(signal, noise_rms, sigma_factor=5, smooth_k=3):
   
    smoothed = moving_avg(signal, k=smooth_k)
    thr = sigma_factor * float(noise_rms)
    candidates = []
    for i in range(1, len(smoothed) - 1):
        s_i, s_prev, s_next = float(smoothed[i]), float(smoothed[i - 1]), float(smoothed[i + 1])
        if s_i >= s_prev and s_i >= s_next and s_i > thr:
            candidates.append(i)
    merged_peaks = merge_peaks_locally(candidates, smoothed, thr)
    return merged_peaks, thr


def find_ref_peaks_on_grid(intens_ref, min_height=0.3, min_distance=2):
   
    peaks = []
    for i in range(1, len(intens_ref) - 1):
        if intens_ref[i] >= intens_ref[i - 1] and intens_ref[i] >= intens_ref[i + 1] and intens_ref[i] >= min_height:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
    return peaks

    




def generate_corrected_labels(
    noisy_spectra,
    original_labels,
    base_names,
    spectra_data,
    molecule_names,
    std_noise,
    sigma_factor=5,
    match_tol=2
):
 
    base_to_ref = {}
    for name, (freqs, intens) in zip(molecule_names, spectra_data):
        base = get_base_molecule_name(name)
        inten = normalize_minmax(intens)
        if base not in base_to_ref:
            base_to_ref[base] = inten
        else:
            base_to_ref[base] = np.maximum(base_to_ref[base], inten)

    corrected = []
    has_array_sigma = isinstance(std_noise, (list, tuple, np.ndarray))

    for idx, spectrum in enumerate(tqdm(noisy_spectra, desc="Correction des labels")):
        y0 = original_labels[idx]
        y = np.zeros_like(y0)
      
        sigma_i = float(std_noise[idx]) if has_array_sigma else (float(std_noise) if std_noise is not None else float(np.std(spectrum)))
        peak_idx, thr = detect_peaks(spectrum, sigma_i, sigma_factor=sigma_factor, smooth_k=3)
        peak_set = set(peak_idx)

        for i, present in enumerate(y0):
            if not present:
                continue
            base = base_names[i]
            ref = base_to_ref.get(base, None)
            if ref is None:
                continue
            ref_peaks = find_ref_peaks_on_grid(ref, min_height=0.3, min_distance=2)
            matches = 0
            for rp in ref_peaks:
                if any((rp + d) in peak_set for d in range(-match_tol, match_tol + 1)):
                    matches += 1
                if matches >= 2:   
                    y[i] = 1
                    break
        corrected.append(y)

    return np.array(corrected, dtype=np.int8)




def generate_synthetic_spectrum(spectra_data, molecule_names, max_molecules):
    indices = np.random.permutation(len(spectra_data))
    used_base_names = set()
    selected_indices = []

    num_to_select = np.random.randint(4, max_molecules + 1)

    for idx in indices:
        full_name = molecule_names[idx]
        base_name = get_base_molecule_name(full_name)
        if base_name not in used_base_names:
            used_base_names.add(base_name)
            selected_indices.append(idx)
        if len(selected_indices) >= num_to_select:
            break

    selected_spectra = [spectra_data[i] for i in selected_indices]
    freqs = selected_spectra[0][0]               
    synthetic_spectrum = np.zeros_like(selected_spectra[0][1], dtype=float)

    for _, intensities in selected_spectra:
        inten = normalize_minmax(intensities)
        inten = shift_spectrum(inten, np.random.randint(-2, 3)) 
        weight = np.random.uniform(0.5, 1.0)
        synthetic_spectrum += weight * inten

    
    synthetic_spectrum = shift_spectrum(synthetic_spectrum, np.random.randint(-3, 4))
    synthetic_spectrum = normalize_minmax(synthetic_spectrum)
    molecule_names_present = [molecule_names[i] for i in selected_indices]

    return synthetic_spectrum, molecule_names_present, freqs



def generate_multiple_spectra(spectra_data, molecule_names, num_spectra, max_molecules):
    all_base_names = sorted(set(get_base_molecule_name(name) for name in molecule_names))
    clean_spectra, spectra_with_noise, spectra_with_pollution = [], [], []
    labels, sigma_list = [], []

    for _ in tqdm(range(num_spectra), desc="Génération"):
        clean, selected_molecules, freqs = generate_synthetic_spectrum(
            spectra_data, molecule_names, max_molecules
        )
        noisy, sigma = add_gaussian_noise(clean)
        polluted = add_pollution(noisy, frac=0.02, amp_sigma=0.6)

        label = np.zeros(len(all_base_names), dtype=int)
        for mol_name in selected_molecules:
            base_name = get_base_molecule_name(mol_name)
            if base_name in all_base_names:
                label[all_base_names.index(base_name)] = 1

        clean_spectra.append(clean)
        spectra_with_noise.append(noisy)
        spectra_with_pollution.append(polluted)
        labels.append(label)
        sigma_list.append(sigma)

    return {
        "clean": np.array(clean_spectra, dtype=np.float32),
        "noise": np.array(spectra_with_noise, dtype=np.float32),
        "pollution": np.array(spectra_with_pollution, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int8),
        "base_names": all_base_names,
        "freqs": freqs,              
        "sigma": np.array(sigma_list, dtype=np.float32),
    }

