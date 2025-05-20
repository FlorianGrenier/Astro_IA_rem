import numpy as np
from .spectral_analysis.analysis import process_spectrum_segments
from .spectral_analysis.treatement import normalize_spectrum, estimate_noise_from_empty_channels, detect_peaks
import matplotlib.pyplot as plt
import os
from .spectral_analysis.utils import load_spectrum_observed, load_spectrum_model
import pandas as pd







spectre_obs = load_spectrum_observed('src/data/raw/complete_spectrum_G34_v_Beff.dat')


# print(spectrum)

# plt.figure(figsize=(10, 5))
# plt.plot(spectrum[:,0], spectrum[:,1], label='Spectre G34 v Beff ', color='blue')
# plt.xlabel('Fréquence (ou Longueur d\'onde)')
# plt.ylabel('Intensité')
# plt.title('Spectre G34 v Beff (nettoyé)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


   
molecule_file_path = {'CH3OH': 'src/data/molecule/Model_CH3OH.dat', # méthanol
                      'Formamide': 'src/data/molecule/Model_formamide.dat' # formamide
                      # 'CH3COCH3': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/acetone.txt', # acétone
                      # 'C2H5OH': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/ethanol.txt', # éthanol
                      # 'C2H5CN': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/ecian.txt', # ethyl cyanide
                      # 'n-C3H7CN': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/pcia.txt', # propyl cyanide
                      # 'C2H3CN': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/vcian.txt', # vinyl cyanide
                      # 'CH3OCHO': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/mf.txt', # methyl formare
                      # 'CH3OCH3': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/dme.txt', # dimethyl ether
                      # 'CH2(OH)CHO': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/ga.txt', # glycolaldehyde
                      # 'NH2CN': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/nh2cn.txt', # cyanamide
                      # 'CH3CHO': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/ch3cho.txt', # acetaldehyde
                      # 'CH2NH': '/net/cremi/flgrenier/espaces/travail/STAGE_ANALYSIS/Astro_IA/src/data/molecule/ch2nh.txt', # methanimine

                      } 


# print(freq_model)


# plt.figure(figsize=(16, 8))
# plt.plot(spectre_model[:,0], spectre_model[:,1], label=f'Molécule CH3OH', color='red')
# plt.xlabel('Fréquence')
# plt.ylabel('Intensité')
# plt.title(f'Modèle CH3OH')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



summary_scores = []


for nom_molecule, chemin_fichier in molecule_file_path.items():

    spectre_model = load_spectrum_model(chemin_fichier)

    results, all_matched_pairs, binary_results, binary_obs, binary_model = process_spectrum_segments(
        spectre_obs, spectre_model, num_segments= 1, nom_molecule = nom_molecule)

    summary_score = {
        'Nom_molécule': nom_molecule,
        'binary_results': binary_results,
    }

    summary_scores.append(summary_score)

    print(f"\nRésultats pour la molécule {nom_molecule}:")

    # df_matches = pd.DataFrame(all_matched_pairs)
    # df_matches.to_csv(f'resultat_chaque_pics_{nom_molecule}.csv', index=False)
    # print(f"Fichier CSV 'resultat_chaque_pics_{nom_molecule}.csv' généré avec succès.")

  
    # df_results = pd.DataFrame(results)
    # df_results.to_csv(f'résultats_segments_{nom_molecule}.csv', index=False)
    # print(f"Fichier CSV 'résultats_segments_{nom_molecule}.csv' généré avec succès.")

    
    # df_binary_results = pd.DataFrame(binary_results)
    # df_binary_results.to_csv(f'résultats_binaire_{nom_molecule}.csv', index=False)
    # print(f"Fichier CSV 'résultats_binaire_{nom_molecule}.csv' généré avec succès.")

    print("Vecteur binaire observé : ", binary_obs)
    print("Vecteur binaire modélisé : ", binary_model)


df_summary = pd.DataFrame(summary_scores)
df_summary.to_csv('résumé_scores.csv', index=False)
print("\nFichier 'résumé_scores.csv' généré avec succès.")


