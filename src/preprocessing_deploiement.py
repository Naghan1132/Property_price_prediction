def preprocessing(df):
    import pandas as pd 
    import pickle as pkl


    ##########
    df.columns = df.columns.str.replace(' ', '_')
    df['Commune'] = df['Commune'].str.replace('-', ' ') 
    df['Commune'] = df['Commune'].str.replace('ST', 'SAINT')  # format 'ST' et 'SAINT'
    df['Commune'] = df['Commune'].str.replace('\'', ' ')

    with open('../data/geoloc_commune.pkl', 'rb') as geo_commune:
        geo_commune = pkl.load(geo_commune)

    # MERGE
    df = pd.merge(df, geo_commune, on=['Commune', 'Code_postal'], how='left')
   


    # Ajouter variable 'niveau de vie par commune'
    with open('../data/niveau_vie.pkl','rb') as niveau_vie:
        niveau_vie = pkl.load(niveau_vie)
    # Ajouter un zéro devant les valeurs de la colonne "Code_postal" si leur longueur est de 4
    df['code_commune_INSEE'] = df['code_commune_INSEE'].astype(str).str.zfill(5)
    # merge sur le code commune INSEE
    df = pd.merge(df, niveau_vie, on=['code_commune_INSEE'], how='left')



    # Créer la variable "day", "month" and "year" 

    df['Date_mutation'] = pd.to_datetime(df['Date_mutation'], format='%d/%m/%Y')

    # Extract the year component and create a new 'year' column
    df['day'] = df['Date_mutation'].dt.day
    df['day'] = df['day'].astype(object)

    df['month'] = df['Date_mutation'].dt.month
    df['month'] = df['month'].astype(object)

    df['year'] = df['Date_mutation'].dt.year
    df['year'] = df['year'].astype(object)

    df = df.drop("Date_mutation", axis=1)



    ## Attribuer un code type local à chaque type et créer une variable
    # Créer un dictionnaire de mapping entre 'Type_local' et 'Code_type_local'
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




    # Imputer
    with open('../model/imputer.pkl', 'rb') as fichier_pickler:
        imputer = pkl.load(fichier_pickler)

    # Utilisation de l'imputer pour les valeurs manquantes
    colonnes_a_imputer = ["Surface_reelle_bati",
                          "Nombre_pieces_principales", "Surface_terrain", "niveau_vie_commune"]
    
    df_2 = df[colonnes_a_imputer]
    df_2_imputed = imputer.transform(df_2)

    df_1 = df.drop(columns=colonnes_a_imputer)

    # Créer un DataFrame à partir du tableau NumPy imputé
    df_2_imputed = pd.DataFrame(df_2_imputed, columns=colonnes_a_imputer)
    df_final = pd.concat([df_1, df_2_imputed], axis=1)





    # Ajouter la variable 'prix au m2 par commune'
    with open('../data/prix_moyen_m2.pkl','rb') as prix_moyen_m2:
        groupe_commune = pkl.load(prix_moyen_m2)

    df_final = pd.merge(df_final, groupe_commune[['Prix_moyen_m2']],
              left_on='Commune', right_index=True, how='left')

    median_value = df_final['Prix_moyen_m2'].median()

    # Remplacez les NaN par la médiane
    import numpy as np
    df_final['Prix_moyen_m2'] = df_final['Prix_moyen_m2'].replace(
        [np.inf, -np.inf], np.nan)
    df_final['Prix_moyen_m2'].fillna(median_value, inplace=True)

    df_final['latitude'] = df_final['latitude'].replace(
        [np.inf, -np.inf], np.nan)
    df_final['latitude'].fillna(median_value, inplace=True)

    df_final['longitude'] = df_final['longitude'].replace(
        [np.inf, -np.inf], np.nan)
    df_final['longitude'].fillna(median_value, inplace=True)

    # PROBLeme si dans le fichier il 'ny a pas la commune etc...et que la personne veut prédire juste un bien  !!
    return (df_final)



    




