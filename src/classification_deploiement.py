def classification(df):

    import pandas as pd 
    import pickle as pkl
    import numpy as np

    # Classification

    mapping_type_local = {
    'Maison': 1,
    'Appartement': 2,
    'Dépendance': 3,
    'Local industriel. commercial ou assimilé': 4
    }

    # Appliquer le mapping pour créer la nouvelle colonne 'Code_type_local'
    df['Code_type_local'] = df['Type_local'].map(mapping_type_local)


    # Remplacer les valeurs manquantes dans la colonne 'Code_type_local' par une valeur par défaut (ici, 0)
    df['Code_type_local'].fillna(0, inplace=True)
    df['Code_type_local'] = df['Code_type_local'].astype(int)


    #Ouverture du modele de classif
    with open('../model/modele_classification.pkl','rb') as fichier_pickler:
        model_classif = pkl.load(fichier_pickler)
    
    # Faire classif sur Code Type Local
    lignes_0 = df[df['Code_type_local'] == 0]
    colonne_classif=['Surface_reelle_bati', 'Nombre_pieces_principales','Surface_terrain']
    predictions_classif = model_classif.predict(lignes_0[colonne_classif])

    #Remplacer les valeurs manquantes dans 'Code_type_local' par les prédictions
    df.loc[df['Code_type_local'] == 0, 'Code_type_local'] = predictions_classif
    


    


    

    