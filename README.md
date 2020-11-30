# Alphabet - Nuage Compta


## Introduction

La société Nuage est une société éditrice d'un logiciel spécialisé dans la gestion de la comptabilité pour les entreprises de toutes tailles : Nuage Compta.

Suite a un sondage auprès de ses clients sur les principales améliorations qu'ils souhaitaient voir intégrer dans le logiciel, la société souhaite développer un modèle de reconnaisance de texte appliqué aux factures de ses clients. Le but de ce modèle serait les aider dans la saisie comptable des pièces de type facture.

La société ne possède pas les ressources nécessaires pour développer un tel module et vous sollicite afin de l'aider à créer ce système.


## Pré-requis

Import des frameworks de Deep Learning Keras et/ou Tensorflow pour entrainer et utiliser notre modèle CNN.

Import des bibliothèques Imutils et CV2 afin d'utiliser notre OCR.

Utilisation du jeu de données fourni.

Utilisation d'un IDE capable de lire et dexécuter les fichiers NoteBook (.ipynb).


## Installation

Pour fonctionner, les fichiers cnn.ipynb et ocr.handwriting.py ont besoin de la structure de projet suivante :

- Un dossier "data" à la racine du projet

- Le dossier "data" doit contenir les quatres dossiers suivants : "RAW", "TEST", "TRAINING" et "VALIDATION".

- Les dossiers "TEST" et "TRAINING" doivent contenir chacun l'ensemble du dataset, c'est à dire 26 dossiers chacun (Un par lettre de l'alphabet).

- Le dossier "VALIDATION" contient des images permettant de valider l'apprentissage de notre modèle CNN.


## Démarrage

Avant de lancer les différents scripts de création et d'entrainement du modèle il faut préparer notre jeu de données comme indiqué précédemment (Dossiers "data/TEST", "data/TRAINING" et "data/VALIDATION").

Pour démarrer le projet, il suffit d'ouvrir le fichier "cnn.ipynb" et d'exécuter les cellules dans l'ordre. Le modèle créé sera directement enregistré à la racine du projet sous le nom "alphabet_model.h5".

Pour utiliser le fichier "ocr_handwriting.py" il suffit de l'exécuter. il utilisera automatiquemennt le modèle nommé "alphabet_model.h5".


## Auteurs

Issam Bouzidi
Maxime Veysseyre