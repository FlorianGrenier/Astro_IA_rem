from .treatement import expand_binary_vector, tanimoto_coefficient, pearson_correlation, jaccard_similarity, match_peaks, normalize_spectrum, ratio_peaks_model, correlate_peak_frequencies, binary_peaks, calculate_frequency_similarity, refine_all_to_centers_aligned

from .treatement import estimate_noise_from_empty_channels, detect_peaks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d


def process_spectrum_segments(spectrum, spectre_model, num_segments, nom_molecule):

    segment_length = len(spectrum) // num_segments
    

    
    all_matched_pairs = []
    results = []
    binary_results = []

    segments_obs = np.array_split(spectrum, num_segments)
    for i, segment in enumerate(segments_obs):


        
        print(f"Segment {i + 1}/{num_segments} : Traitement du segment.")
       
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < num_segments - 1 else len(spectrum)

        freq_segment_obs = spectrum[start_idx:end_idx, 0]
        intensity_segment_obs = spectrum[start_idx:end_idx, 1]

       
        mask_model = (spectre_model[:, 0] >= freq_segment_obs[0]) & (spectre_model[:, 0] <= freq_segment_obs[-1])
        freq_segment_model = spectre_model[mask_model, 0]
        intensity_segment_model = spectre_model[mask_model, 1]


      


        

        delta_f_obs = np.min(np.diff(freq_segment_obs))
        delta_f_model = np.min(np.diff(freq_segment_model))

        
        delta_f_common = min(delta_f_obs, delta_f_model)

     
        f_min = max(min(freq_segment_obs), min(freq_segment_model))
        f_max = min(max(freq_segment_obs), max(freq_segment_model))

       
        common_freq = np.arange(f_min, f_max, delta_f_common)

        interp_obs = interp1d(freq_segment_obs, intensity_segment_obs, kind='linear', bounds_error=False, fill_value=0)
        interp_model = interp1d(freq_segment_model, intensity_segment_model, kind='linear', bounds_error=False, fill_value=0)

        s_obs_interp = interp_obs(common_freq)
        s_model_interp = interp_model(common_freq)

        mask = s_obs_interp > 0
        common_freq = common_freq[mask]
        s_obs_interp = s_obs_interp[mask]


       

        # Normalisation du spectre observé

        intensity_segment_obs_norm = normalize_spectrum(s_obs_interp)
        intensity_segment_model_norm = normalize_spectrum(s_model_interp)
        
        

        mean_noise, std_noise = estimate_noise_from_empty_channels(intensity_segment_obs_norm) 



       
        peak_indices_obs, threshold, sigma_factor = detect_peaks(intensity_segment_obs_norm, std_noise, sigma_factor=5)

        
        peak_freqs_obs = common_freq[peak_indices_obs]

    

        plt.figure(figsize=(10, 5))
        plt.plot(common_freq, intensity_segment_obs_norm, label='Signal')
        plt.plot(common_freq[peak_indices_obs], intensity_segment_obs_norm[peak_indices_obs], 'ro', label='Pics détectés')
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Seuil ({sigma_factor} sigma)')

        plt.title("Détection de pics")
        plt.xlabel("Fréquence (MHz)")
        plt.ylabel("Amplitude normalisée")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        mean_noise, std_noise = estimate_noise_from_empty_channels(intensity_segment_model_norm) 

     
        peak_indices_model, threshold, sigma_factor = detect_peaks(intensity_segment_model_norm, std_noise, sigma_factor=5)
        peak_freqs_model = common_freq[peak_indices_model]

       
        print(f"Fréquences des pics modélisés : {peak_freqs_model}")
       

        plt.figure(figsize=(10, 5))
        plt.plot(common_freq, intensity_segment_model_norm, label='Signal')
        plt.plot(common_freq[peak_indices_model], intensity_segment_model_norm[peak_indices_model], 'ro', label='Pics détectés')
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Seuil ({sigma_factor} sigma)')

        plt.title("Détection de pics")
        plt.xlabel("Fréquence (MHz)")
        plt.ylabel("Amplitude normalisée")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        

   

       
        max_corr, shift_ghz = correlate_peak_frequencies(
            common_freq, peak_freqs_obs, peak_freqs_model
        )


        print(f"Correlation max : {max_corr}")

        print(f"Décalage (MHz) : {shift_ghz:.6f}")

        # Appliquer le décalage au spectre modélisé
        # freq_segment_model_shifted = freq_segment_obs + shift_ghz
        # intensity_segment_model_norm = freq_segment_model(freq_segment_model_shifted)
        


    #     # Prendre les pics du modèle alignés
        # print("Spectre aligné")
        mean_noise, std_noise = estimate_noise_from_empty_channels(intensity_segment_model_norm)
        
        peak_indices_model_aligné, threshold, sigma_factor  = detect_peaks(intensity_segment_model_norm, std_noise, sigma_factor=5)
        peak_freqs_model_aligné = common_freq[peak_indices_model_aligné]

    #     # print(f"Intensité pics modélisés : {intensity_segment_model_norm[peak_indices_model_aligné]}")

    #     # plt.figure(figsize=(10, 5))
    #     # plt.plot(common_freq, intensity_segment_model_norm, label='Signal')
    #     # plt.plot(common_freq[peak_indices_model], intensity_segment_model_norm[peak_indices_model], 'ro', label='Pics détectés')
    #     # plt.axhline(y=threshold, color='green', linestyle='--', label=f'Seuil ({sigma_factor} sigma)')

    #     # plt.title("Détection de pics")
    #     # plt.xlabel("Fréquence (GHz)")
    #     # plt.ylabel("Amplitude normalisée")
    #     # plt.legend()
    #     # plt.grid(True)
    #     # plt.tight_layout()
    #     # plt.show()



    #     # Créer des vecteurs binaires pour les pics
        binary_vector_obs = binary_peaks(peak_indices_obs, common_freq)
        binary_vector_model = binary_peaks(peak_indices_model_aligné, common_freq)

        binary_vector_obs = expand_binary_vector(binary_vector_obs, common_freq, MHz_window=2.0)
        binary_vector_model = expand_binary_vector(binary_vector_model, common_freq, MHz_window=2.0)

        binary_vector_obs, binary_vector_model = refine_all_to_centers_aligned(binary_vector_obs, binary_vector_model)


        print(f"Vecteur binaire observé : {binary_vector_obs}")
        print(f"Vecteur binaire modélisé : {binary_vector_model}")





       
  



       
        cm = confusion_matrix(binary_vector_obs, binary_vector_model)
        cm_display = ConfusionMatrixDisplay(cm, display_labels=['Non Pic', 'Pic'])
        cm_display.plot(cmap=plt.cm.Blues)
        plt.title(f"Matrice de confusion - Segment {i + 1}")
        plt.xlabel("Prédictions")
        plt.ylabel("Vérités de terrain")
        plt.grid(False)
        plt.show()
     
        precision = precision_score(binary_vector_obs, binary_vector_model)
        recall = recall_score(binary_vector_obs, binary_vector_model)
     
        print(f"Précision : {precision:.4f}")
        print(f"Rappel : {recall:.4f}")

        f1 = f1_score(binary_vector_obs, binary_vector_model)
        print(f"F1-score : {f1:.4f}")

        binary_result = {
            'Nom_molécule': nom_molecule,
            'Segment': i + 1,

            'Précision': precision,
            'Rappel': recall,
            'F1-score': f1
            
        }
        binary_results.append(binary_result)

        

        

      
  
        
    #     # Calculer la similarité entre les fréquences des pics matchés observées et modélisées

       
        
        matched_pairs = match_peaks(peak_freqs_obs, peak_freqs_model_aligné, tol_MHz=2.0)
       


        print(f"Paires de pics matchés ({len(matched_pairs)}):")
        nb_matched = len(matched_pairs)
        nb_obs = len(peak_freqs_obs)
        nb_model = len(peak_freqs_model_aligné)

      
        recall = nb_matched / nb_model if nb_model > 0 else 0
    

        for f_obs, f_model in matched_pairs:
            sim_freq = calculate_frequency_similarity(f_obs, f_model)
            print(f"Pic observé à {f_obs:.6f} MHz ↔ modèle à {f_model:.6f} MHz")
            print(
                f"SAM = {sim_freq['SAM_cosine']:.6f}, "
                f"Ratio = {sim_freq['Ratio']:.6f}, "
                f"Euclidean_distance = {sim_freq['Euclidean_distance']:.6f}"
               
            )
            all_matched_pairs.append({
                'Segment': i + 1,
                'Décalage (MHz)': shift_ghz,
                'Pic observé (MHz)': f_obs,
                'Pic modèle aligné (MHz)': f_model,
                'SAM': sim_freq['SAM_cosine'],
                'Ratio fréquence': sim_freq['Ratio'],
                'Euclidean_distance': sim_freq['Euclidean_distance'] 
            })

        
        segment_result = {
            'Segment': i + 1,
            'Décalage (MHz)': shift_ghz,
            'Corrélation max': max_corr,
            'Nb pics obs': len(peak_freqs_obs),
            'Nb pics modèle aligné': len(peak_freqs_model_aligné),
            'Nb pics matchés': len(matched_pairs),
            'Ratio pics matchés': ratio_peaks_model(matched_pairs, peak_freqs_model_aligné),
         
            'Rappel': recall,
            'Tanimoto coefficient': tanimoto_coefficient(binary_vector_obs, binary_vector_model),
            'Pearson': pearson_correlation(binary_vector_obs, binary_vector_model),
            'Jaccard': jaccard_similarity(binary_vector_obs, binary_vector_model)
        }
        results.append(segment_result)

        print(f"Ratio des pics du modèle : {ratio_peaks_model(matched_pairs, peak_freqs_model_aligné):.6f}")

        print(f"Prédiction (Nombre de pics modele bien prédit): {recall}")

        tanimoto_coefficient_value = tanimoto_coefficient(binary_vector_obs, binary_vector_model)
        print(f"Tanimoto coefficient : {tanimoto_coefficient_value:.6f}")


        pearson_value = pearson_correlation(binary_vector_obs, binary_vector_model)
        print(f"Pearson correlation : {pearson_value:.6f}")
        
        jaccard_similarity_value = jaccard_similarity(binary_vector_obs, binary_vector_model)
        print(f"Jaccard similarity : {jaccard_similarity_value:.6f}")
        

  
        

        
    

        plt.figure(figsize=(10, 4))
        plt.plot(common_freq, intensity_segment_obs_norm, label='Spectre Observé', color='blue')
        plt.plot(common_freq, intensity_segment_model_norm, label='Spectre Modèle', color='red')
        for f_obs, f_model in matched_pairs:
            plt.axvline(f_obs, color='blue', linestyle='--', alpha=0.5)
            plt.axvline(f_model, color='red', linestyle='--', alpha=0.5)
        plt.title(f'Spectres - Segment {i + 1}')
        plt.xlabel('Fréquence (GHz)')
        plt.ylabel('Intensité')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()







        plt.figure(figsize=(10, 4))
        plt.plot(common_freq, binary_vector_obs, label='Pics Observés', color='blue', alpha=0.5)
        plt.plot(common_freq, binary_vector_model , label='Pics Modèle ', color='red', alpha=0.5)
        plt.title(f'Pics - Segment {i + 1}')
        plt.xlabel('Fréquence (GHz)')
        plt.ylabel('Présence de pic (0 ou 1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



   
   
    return results, all_matched_pairs, binary_results, binary_vector_obs, binary_vector_model




