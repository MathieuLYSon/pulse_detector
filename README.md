# Détection du pouls humain à partir de vidéos du visage

## Objectif du projet

Projet de détection et estimation de la fréquence cardiaque humaine à partir de vidéos faciales synchronisées avec des enregistrements ECG.

Ce projet vise à développer un algorithme de **détection du pouls humain à partir de vidéos (rPPG)** (Remote Photoplethysmography), en s’appuyant sur l’analyse des variations de couleur de la peau. Il permet également la comparaison et l’alignement précis entre les signaux rPPG et ECG.

L'algorithme extrait le signal à partir de vidéos en analysant les variations de couleur de la peau sur les joues, synchronise ces signaux avec l'ECG via corrélation croisée et compare la performance des deux sources.

## Dataset du MIT

Dateset utilisé : https://github.com/vladostan/Dataset-for-video-based-pulse-detection

Open dataset for video-based heart rate estimation. Includes .mp4 video files and ground truth ECG signals 
- Videos
	> 20 sec video fragments in two physical conditions: in rest and after performing physical exercises
- ECG
	> 20 sec ECGs recorded using [PC based 6-lead Resting ECG/EKG Workstation](https://vdd-pro.ru/en/product/pc-based-6-lead-resting-ecg-ekg-workstationanalysis-software-kopirovat/) in .cardio and .txt
	- .cardio - Can be opened using [ECG Control](https://vdd-pro.ru/en/2014/03/ecg-control-user-manual/) software
	- .txt - Contains six signals from six leads (I, II, III, avR, avL, avF)


Chaque patient possède deux conditions testées : `normal` et `physical`.

## Méthodologie

- **Extraction de la ROI faciale** : Joues extraites automatiquement via `MediaPipe FaceMesh` (landmarks 205 et 425).
- **Extraction du signal rPPG** :
  - Moyenne des couleurs RGB sur la ROI.
  - PCA pour séparer les composantes pulsatives.
  - Filtrage passe-bande (cardiaque).
  - Estimation de la fréquence via détection de pics et FFT.
- **Synchronisation rPPG/ECG** :
  - Comparaison des durées d’enregistrement.
  - Alignement temporel par cross-correlation des signaux.
- **Comparaison** :
  - Comparaison des BPM prédits pour les signaux rPPG et ECG après synchronisation.

## Exécution

Depuis le Terminal:

# Créer l'environnement définit dans environment.yml
conda env create -f environment.yml
# Activer l'environnement
conda activate rppg_env
# Ouvrir le Jyputer notebook
jupyter notebook

Le projet est entièrement contenu dans un Notebook `hr_estimation.ipynb` exécutable dans l'environnement définit dans le fichier **environment.yml**.  
Il nécessite la présence des vidéos et fichiers ECG dans l'arborescence `data/`.