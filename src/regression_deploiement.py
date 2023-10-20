def regression(df):
    import pickle as pkl

    # Regression
    with open('../model/modele_regression_code_1.pkl', 'rb') as fichier_pickler:
        model_reg_1 = pkl.load(fichier_pickler)

    with open('../model/modele_regression_code_2.pkl', 'rb') as fichier_pickler:
        model_reg_2 = pkl.load(fichier_pickler)

    with open('../model/modele_regression_code_3.pkl', 'rb') as fichier_pickler:
        model_reg_3 = pkl.load(fichier_pickler)

    with open('../model/modele_regression_code_4.pkl', 'rb') as fichier_pickler:
        model_reg_4 = pkl.load(fichier_pickler)

    

    def predict_valeur_fonciere(row):

        col_pred = ['Surface_reelle_bati', 'Nombre_pieces_principales', 'Surface_terrain',
       'month', 'latitude', 'longitude', 'niveau_vie_commune',
       'Prix_moyen_m2']

        # ajouter lat et long et niveau de vie à sub_final 
        if row['Code_type_local'] == 1:
            pred = model_reg_1.predict(row[col_pred].values.reshape(1, -1))
            return pred
        elif row['Code_type_local'] == 2:
            pred = model_reg_2.predict(row[col_pred].values.reshape(1, -1))
            return pred
        elif row['Code_type_local'] == 3:
            pred = model_reg_3.predict(row[col_pred].values.reshape(1, -1))
            return pred
        elif row['Code_type_local'] == 4:
            pred = model_reg_4.predict(row[col_pred].values.reshape(1, -1))
            return pred


    # Appliquer la fonction à chaque ligne du DataFrame
    df['TARGET'] = df.apply(predict_valeur_fonciere, axis=1)

    # Mise en forme des prédictions
    def convertir(liste):
        return liste[0]

    # Appliquer la fonction à la colonne 'colonne_liste' à l'aide de apply() et d'une fonction lambda
    df['TARGET'] = df['TARGET'].apply(lambda x: convertir(x))
    last_cols = ['ID','TARGET']
    df = df[last_cols]
    df.set_index('ID', inplace=True)
    return (df)
