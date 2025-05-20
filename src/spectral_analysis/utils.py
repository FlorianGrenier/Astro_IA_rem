
import numpy as np
import matplotlib.pyplot as plt


## charge le spectre observé et le modèle


def load_spectrum_observed(name_spectrum):
    """
    Charge le spectre observé à partir d'un fichier.
    """
    
   
    file_path = name_spectrum
    spectre_obs_data = np.loadtxt(file_path)



    x = spectre_obs_data[:, 0] 
    y = spectre_obs_data[:, 1] 


    mask = y > 0
    x_clean = x[mask]
    y_clean = y[mask]


    spectre_obs = np.column_stack((x_clean, y_clean))

    return spectre_obs

def load_spectrum_model(name_spectrum):
    """
    Charge le spectre du modèle à partir d'un fichier.
    """
   
    molecule_file_path =  name_spectrum
  

    try:
        molecule_data = np.loadtxt(molecule_file_path)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {molecule_file_path}: {e}")
        return None
       


    freq_model = molecule_data[:, 0]
    intensity_model = molecule_data[:, 1]

    spectre_model = np.column_stack((freq_model, intensity_model))

    return spectre_model