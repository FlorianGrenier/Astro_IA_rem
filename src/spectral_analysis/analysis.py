from .treatement import normalize_spectrum, get_peak, ratio_peaks_model, correlate_peak_frequencies, binary_peaks, calculate_frequency_similarity

import matplotlib.pyplot as plt
import pandas as pd

def process_spectrum_segments(spectre_obs, spectre_model, num_segments):
    segment_length = len(spectre_obs) // num_segments
    results = []
    all_matched_pairs = []

    for i in range(num_segments):
        print(f"Segment {i + 1}/{num_segments} : Traitement du segment.")

        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < num_segments - 1 else len(spectre_obs)
        
        freq_segment_obs = spectre_obs[start_idx:end_idx, 0]
        intensity_segment_obs = spectre_obs[start_idx:end_idx, 1]


        mask_model = (spectre_model[:, 0] >= freq_segment_obs[0]) & (spectre_model[:, 0] <= freq_segment_obs[-1])
        freq_segment_model = spectre_model[mask_model, 0]
        intensity_segment_model = spectre_model[mask_model, 1]

        intensity_segment_obs_norm = normalize_spectrum(intensity_segment_obs)
        intensity_segment_model_norm = normalize_spectrum(intensity_segment_model)

        peak_indices_obs, peak_freqs_obs = get_peak(freq_segment_obs, intensity_segment_obs_norm)
        peak_indices_model, peak_freqs_model = get_peak(freq_segment_model, intensity_segment_model_norm)

        max_corr, shift_ghz = correlate_peak_frequencies(
            freq_segment_obs,  peak_freqs_obs, peak_freqs_model
        )


        # Appliquer le décalage aux fréquences du modèle
        freq_segment_model += shift_ghz

        # Prendre les pics du modèle alignés
        print("Spectre aligné")
        peak_indices_model_aligné, peak_freqs_model_aligné = get_peak(freq_segment_model, intensity_segment_model_norm)



        # Créer des vecteurs binaires pour les pics
        binary_vector_obs = binary_peaks(peak_indices_obs, freq_segment_obs)
        binary_vector_model =  binary_peaks(peak_indices_model_aligné, freq_segment_model)
       

        # Calculer la similarité entre les fréquences des pics matchés observées et modélisées
        

        matched_pairs = []
        for f_model in peak_freqs_model_aligné:
            for f_obs in peak_freqs_obs:
                if abs(f_model - f_obs) <= 10:
                    matched_pairs.append((f_obs, f_model))
                    break  
        
        print(f"Paires de pics matchés ({len(matched_pairs)}):")
        for f_obs, f_model in matched_pairs:
            sim_freq = calculate_frequency_similarity(f_obs, f_model)
            print(f"Pic observé à {f_obs:.6f} GHz ↔ modèle à {f_model:.6f} GHz")
            print(
                f"SAM = {sim_freq['SAM_cosine']:.6f}, "
                f"Ratio = {sim_freq['Ratio']:.6f}, "
               
            )
            all_matched_pairs.append({
                'Segment': i + 1,
                'Décalage (GHz)': shift_ghz,
                'Pic observé (GHz)': f_obs,
                'Pic modèle aligné (GHz)': f_model,
                'SAM': sim_freq['SAM_cosine'],
                'Ratio fréquence': sim_freq['Ratio']
            })

        
        segment_result = {
            'Segment': i + 1,
            'Décalage (GHz)': shift_ghz,
            'Corrélation max': max_corr,
            'Nb pics obs': len(peak_freqs_obs),
            'Nb pics modèle aligné': len(peak_freqs_model_aligné),
            'Nb pics matchés': len(matched_pairs),
            'Ratio pics matchés': ratio_peaks_model(matched_pairs, peak_freqs_model_aligné)
        }
        results.append(segment_result)

        print(f"Ratio des pics du modèle : {ratio_peaks_model(matched_pairs, peak_freqs_model_aligné):.6f}")
        
    

        # Tracer les spectres observés et modélisés
        plt.figure(figsize=(10, 4))
        plt.plot(freq_segment_obs, intensity_segment_obs, label='Spectre Observé', color='blue')
        plt.plot(freq_segment_model, intensity_segment_model, label='Spectre Modèle', color='red')
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







        # Tracer les pics binaires
        plt.figure(figsize=(10, 4))
        plt.plot(freq_segment_obs, binary_vector_obs, label='Pics Observés', color='blue', alpha=0.4)
        plt.plot(freq_segment_model, binary_vector_model , label='Pics Modèle ', color='red', alpha=0.6)
        plt.title(f'Pics - Segment {i + 1}')
        plt.xlabel('Fréquence (GHz)')
        plt.ylabel('Présence de pic (0 ou 1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        


        


   
    df_matches = pd.DataFrame(all_matched_pairs)
    df_matches.to_csv('correspondances_pics.csv', index=False)
    print("Fichier CSV 'correspondances_pics.csv' généré avec succès.")

    df_results = pd.DataFrame(results)
    df_results.to_csv('résultats_segments.csv', index=False)
    print("Fichier CSV 'résultats_segments.csv' généré avec succès.")


  


