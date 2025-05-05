# Astro_IA

AI for astronomical spectral line analysis.

## Structure du projet


## Description des dossiers

- **astro_ia/** : Contient les fichiers et bibliothèques nécessaires à l'environnement Python virtuel.
- **output/** : Contient les fichiers de sortie générés par les analyses, tels que des graphiques et des résultats intermédiaires.
- **src/** : Contient le code source principal du projet.
  - `main.py` : Point d'entrée principal du projet.
  - `spectral_analysis/` : Module pour l'analyse spectrale, incluant des fonctions comme `process_spectrum_segments`.
  - `data/` : Contient les fichiers de données brutes et de modèles utilisés pour les analyses.
  - `result_correspondance/` : Stocke les résultats des correspondances entre pics spectraux.
  - `test/` : Contient des tests pour valider les fonctionnalités.
- **test_methode/** : Contient des notebooks Jupyter pour tester différentes méthodes d'analyse, comme la corrélation croisée et DTW (Dynamic Time Warping).

- **requirements.txt** : Liste des dépendances Python nécessaires au projet.
- **AI_for_spectral_line_analysis.pdf** : Sujet lié au projet.

## Fonctionnalités principales

- Analyse des spectres astronomiques pour détecter et comparer les lignes spectrales.
- Correspondance entre les pics des spectres observés et modélisés.
- Génération de graphiques pour visualiser les spectres et les correspondances.
- Exportation des résultats sous forme de fichiers CSV.

## Installation

1. Clonez ce dépôt :
    ```bash
    git clone <>
    cd Astro_IA

2. Activez l'environnement virtuel :

    source astro_ia.bin/activate

3. Installez les dépendances :

    pip install -r requirements.txt

Utilisation
Placez vos fichiers de données dans le dossier src/data/.
Exécutez le script principal :

1. Placez vos fichiers de données dans le dossier src/data/.

2. Exécutez le script principal :

python -m src/main.py

Auteurs

GRENIER Florian