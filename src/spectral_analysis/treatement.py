import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate



def normalize_spectrum(intensity):
    return (intensity - np.mean(intensity)) / np.std(intensity)

def get_peak(freq, intensity, threshold_sigma=5, smoothing_window=3):
    intensity_smooth = uniform_filter1d(intensity, size=smoothing_window)
    

    sigma = np.std(intensity_smooth)
    threshold = threshold_sigma * sigma


    peak_indices, _ = find_peaks(intensity_smooth, height=threshold)
    peak_indices = np.clip(peak_indices, 0, len(freq) - 1)

  
    peak_freqs = freq[peak_indices]

    plt.figure(figsize=(10, 4))
    plt.plot(freq, intensity, label='Spectre', color='blue', alpha=0.6)
    plt.plot(freq[peak_indices], intensity[peak_indices], 'ro', label='Pics détectés')
    plt.plot(freq, intensity_smooth, label='Spectre lissé', color='orange', alpha=0.6)
    plt.axhline(threshold, color='green', linestyle='--', label='Seuil de détection')
    
    plt.xlabel('Fréquence (GHz)')
    plt.ylabel('Intensité')
    plt.title('Détection des pics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return peak_indices, peak_freqs



def binary_peaks(peak_indices, freq):
    binary_vector = np.zeros_like(freq, dtype=int)
    binary_vector[peak_indices] = 1
    return binary_vector


def correlate_peak_frequencies(freq_obs, peaks_obs, peaks_model):


    if len(peaks_obs) == 0 or len(peaks_model) == 0:
        return 0, 0

    corr = correlate(peaks_obs, peaks_model, mode='full')
    lags = np.arange(-len(peaks_model) + 1, len(peaks_obs))

    max_corr = np.max(corr)
    best_shift = lags[np.argmax(corr)]

    shift_ghz = best_shift * (freq_obs[1] - freq_obs[0]) 

    return max_corr, shift_ghz


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



def ratio_peaks_model(matched_pairs, peak_freqs_model_aligné):
    if len(peak_freqs_model_aligné) == 0 and len(matched_pairs) == 0:
        return 0
    else :
        return len(matched_pairs) / len(peak_freqs_model_aligné)
    




def calculate_frequency_similarity(f_obs, f_model):
    f_obs = float(f_obs)
    f_model = float(f_model)

   
    cosine_sim = spectral_angle_mapper([f_obs], [f_model])
    
    # ratio de la fréquence 
    ratio = f_obs / f_model if f_model != 0 else np.nan

  

    return {
       
        'SAM_cosine': cosine_sim,
        'Ratio': ratio,
       
    }



