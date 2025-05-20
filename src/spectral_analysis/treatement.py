import numpy as np
import matplotlib.pyplot as plt



from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate
from scipy.signal import savgol_filter



# Pretreatment of the spectrum

def normalize_spectrum(intensity):
    intensity = np.nan_to_num(intensity, nan=0.0)  
    if np.all(intensity == intensity[0]): 
        return np.zeros_like(intensity)
    
    mean = np.mean(intensity)
    std = np.std(intensity)
    return (intensity - mean) / std


def estimate_noise_from_empty_channels(intensity, threshold=0.01):
    
    noise_indices = np.where(intensity < threshold)[0]
    
    if len(noise_indices) == 0:
        print("Aucune zone sans signal détectée.")
        return None, None

  
    noise_intensity = intensity[noise_indices]
    
   
    mean_noise = np.mean(noise_intensity)
    std_noise = np.std(noise_intensity)
    
    return mean_noise, std_noise



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

     
        while j < len(candidates) and candidates[j] - current < current_width:
            if signal[candidates[j]] > signal[current]:
                current = candidates[j]
                current_width = compute_local_width(signal, current, threshold)
            j += 1

        merged.append(current)
        i = j

    return merged

def detect_peaks(signal, noise_rms, sigma_factor=5, smoothing_window= 5):

  

    smoothed = savgol_filter(signal, window_length=smoothing_window, polyorder=3)
    # smoothed = uniform_filter1d(signal, size=smoothing_window)

    smoothed = signal
    threshold = sigma_factor * noise_rms

    print(f"Seuil de détection : {threshold:.4f} (sigma = {sigma_factor})")

    candidates = []
    for i in range(1, len(smoothed) - 1):
        if (smoothed[i] > smoothed[i-1] and
            smoothed[i] > smoothed[i+1] and
            smoothed[i] > threshold):
            candidates.append(i)

    merged_peaks = merge_peaks_locally(candidates, smoothed, threshold)

    return merged_peaks, threshold, sigma_factor

   

    

    
def binary_peaks(peak_indices, freq):
    binary_vector = np.zeros_like(freq, dtype=int)
    binary_vector[peak_indices] = 1
    return binary_vector

def expand_binary_vector(binary_vector, freq, MHz_window=2.0):

    expanded_vector = np.zeros_like(binary_vector)
    freq = np.array(freq)
    peak_indices = np.where(binary_vector == 1)[0]

    for idx in peak_indices:
        f_center = freq[idx]
        indices_in_window = np.where(np.abs(freq - f_center) <= MHz_window)[0]
        expanded_vector[indices_in_window] = 1

    return expanded_vector


def group_indices(indices):
    if len(indices) == 0:
        return []
    groups = []
    current_group = [indices[0]]
    for idx in indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)
    return groups


def refine_all_to_centers_aligned(binary_vector_obs, binary_vector_model):
    import numpy as np

    refined_obs = np.zeros_like(binary_vector_obs)
    refined_model = np.zeros_like(binary_vector_model)

    obs_indices = np.where(binary_vector_obs == 1)[0]
    model_indices = np.where(binary_vector_model == 1)[0]

    obs_groups = group_indices(obs_indices)
    model_groups = group_indices(model_indices)

    used_model = set()

    for og in obs_groups:
        matched = False
        for i, mg in enumerate(model_groups):
            if set(og) & set(mg):  # chevauchement
                intersection = sorted(set(og) & set(mg))
                center = intersection[len(intersection) // 2]  # centre de l'intersection
                refined_obs[center] = 1
                refined_model[center] = 1
                used_model.add(i)
                matched = True
                break
        if not matched:
            center_obs = og[len(og) // 2]
            refined_obs[center_obs] = 1

    for i, mg in enumerate(model_groups):
        if i not in used_model:
            center_model = mg[len(mg) // 2]
            refined_model[center_model] = 1

    return refined_obs, refined_model



# Similarity measures

def euclidean_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)


def spectral_angle_mapper(a, b):
    a = np.array(a)
    b = np.array(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return np.pi / 2 

    cos_sim = dot_product / (norm_a * norm_b)
    cos_sim = np.clip(cos_sim, -1.0, 1.0) 

    return np.degrees(np.arccos(cos_sim))




def correlate_peak_frequencies(freq_obs, peaks_obs, peaks_model):
    """Correlate observed and model peak frequencies."""
    if len(peaks_obs) == 0 or len(peaks_model) == 0:
        return 0, 0
    corr = correlate(peaks_obs, peaks_model, mode='full')
    lags = np.arange(-len(peaks_model) + 1, len(peaks_obs))
    max_corr = np.max(corr)
    best_shift = lags[np.argmax(corr)]
    shift_ghz = best_shift * (freq_obs[1] - freq_obs[0])
    return max_corr, shift_ghz


def calculate_frequency_similarity(f_obs, f_model, eps=1e-6):
    f_obs = float(f_obs)
    f_model = float(f_model)

    if abs(f_model) < eps:
        return {
            'SAM_cosine': np.nan,
            'Ratio': np.nan,
            'Euclidean_distance': np.nan
        }

    distance_euclidienne = euclidean_distance([f_obs], [f_model])
    cosine_sim = spectral_angle_mapper([f_obs], [f_model])
    ratio = f_obs / f_model

    return {
        'SAM_cosine': cosine_sim,
        'Ratio': ratio,
        'Euclidean_distance': distance_euclidienne
    }



def ratio_peaks_model(matched_pairs, peak_freqs_model_aligné):
    if len(peak_freqs_model_aligné) == 0 and len(matched_pairs) == 0:
        return 0
    else :
        return len(matched_pairs) / len(peak_freqs_model_aligné)


def tanimoto_coefficient(a, b):
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.dot(a, a)
    norm_b = np.dot(b, b)
    den = norm_a + norm_b - dot_product
    return dot_product / den if den != 0 else 0

def pearson_correlation(a, b):
    a = np.array(a)
    b = np.array(b)

    if len(a) != len(b):
        raise ValueError("Les vecteurs doivent avoir la même longueur.")

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    numerator = np.sum((a - mean_a) * (b - mean_b))
    denominator = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))

    return numerator / denominator if denominator != 0 else 0


def jaccard_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    intersection = np.sum(np.logical_and(a, b))
    union = np.sum(np.logical_or(a, b))

    return intersection / union if union != 0 else 0




def match_peaks(peak_freqs_obs, peak_freqs_model, tol_MHz=2.0):
   
    matched_pairs = []
    used_obs = set()

    for f_model in peak_freqs_model:
        for f_obs in peak_freqs_obs:
            if f_obs in used_obs:
                continue  
            if abs(f_model - f_obs) <= tol_MHz:
                matched_pairs.append((f_obs, f_model))
                used_obs.add(f_obs)
                break 

    return matched_pairs
