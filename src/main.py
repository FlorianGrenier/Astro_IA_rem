import numpy as np
from .spectral_analysis.analysis import process_spectrum_segments






file_path = 'src/data/raw/complete_spectrum_G34_v_Beff.dat'
spectre_obs_data = np.loadtxt(file_path)



x = spectre_obs_data[:, 0] 
y = spectre_obs_data[:, 1] 


mask = y > 0
x_clean = x[mask]
y_clean = y[mask]


spectre_obs = np.column_stack((x_clean, y_clean))

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




molecule_file_path =  'src/data/molecule/Model_CH3OH.dat'
   
 



molecule_data = np.loadtxt(molecule_file_path)



freq_model = molecule_data[:, 0]
intensity_model = molecule_data[:, 1]

spectre_model = np.column_stack((freq_model, intensity_model))

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




process_spectrum_segments(spectre_obs, spectre_model, num_segments=50)





