# Property Price Prediction

Ce projet implémente un modèle de prédiction des prix des biens immobiliers en utilisant Python et des techniques d'apprentissage automatique. Les données utilisées pour l'entraînement du modèle proviennent de [Data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) et couvrent les années 2018, 2019, 2020 et 2021.

## Source des Données

Les données d'entraînement sont extraites du jeu de données public de [Data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) qui contient des informations sur les demandes de valeurs foncières pour les années 2018, 2019, 2020 et 2021.

## Fonctionnalités

- **Prédiction de Prix :** Le modèle utilise des caractéristiques telles que la superficie, le nombre de chambres, l'emplacement, etc., pour prédire le prix des biens immobiliers.

## Comment Utiliser le Projet

1. **Téléchargement des Données :** Téléchargez les données à partir du lien [Data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) et placez le fichier CSV dans le dossier `data/`.

2. **Installation des Dépendances :** Assurez-vous d'avoir installé Python sur votre système. Installez les dépendances en utilisant le fichier `requirements.txt` :


3. **Prétraitement des données :** Pour entraîner le modèle, utilisez le script `train.py` en fournissant les données d'entraînement appropriées.

4. **Classifcation de code type local :** Une fois le modèle entraîné, vous pouvez l'utiliser pour prédire le prix des biens immobiliers en utilisant le script `predict.py` et en fournissant les caractéristiques en entrée.

4. **Régression :** Une fois le modèle entraîné, vous pouvez l'utiliser pour prédire le prix des biens immobiliers en utilisant le script `predict.py` et en fournissant les caractéristiques en entrée.


5. **Prédiction :** Une fois le modèle entraîné, vous pouvez l'utiliser pour prédire le prix des biens immobiliers en utilisant le script `predict.py` et en fournissant les caractéristiques en entrée.


## Dashboard

Le dashboard est accessible à l'adresse suivante : [Dashboard](http://dash.eu-4.evennode.com/)

![Dashboard](https://imgur.com/a/xWkPOab)

![Prédictions](https://imgur.com/a/Ew0qgcF)


## Structure du Projet

- **`data/` :** Ce dossier contient des données additionnelles aux fichiers CSV utilisés.
- **`src/` :** Ce dossier contient le code source du projet, y compris les scripts d'entraînement et de prédiction.
- **`models/` :** Ce dossier contient les modèles entraînés.
- **`requirements.txt` :** Ce fichier contient la liste des dépendances requises pour le projet.

## Auteurs

Nathan GRIMAULT
Ivan
Cyrielle
Joe

## Licence

Ce projet est sous licence [Nom de la Licence]. Consultez le fichier LICENCE pour plus de détails.


