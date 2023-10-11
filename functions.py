import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scientisttools.discriminant_analysis import DISMIX
from sklearn.feature_selection import VarianceThreshold

#will load a dataframe from the root directory
##params:
### sep==> separator in the file
### file=> road-to-file/name-of-file.extension
##return:
### df (as a dataframe)
def read_t(file, sep="|"):
    ext= file.split(".")[-1]
    if ext in ["txt"]:
        df= pd.read_table(file, sep=sep)
    elif ext in ["excel"]:
        df= pd.read_excel(file, sep=sep)
    return df


#Allow univariate analysis 
## version 1: pour la premiere analyse
## version 2: pour la deuxieme analyse
def univariate_analysis(df, version=1):
    if version==1:
        print("la taille de données: ")
        print(f"la base a {df.shape[0]} lignes et {df.shape[1]} variables")
        print("Description statistique de nos données: ")
        display(df.describe())
        print("\n")
        print("\n")
        print("Séparation des données par type: ")
        infos= {str(k):list(v.columns) for k,v in df.groupby(df.dtypes, axis=1)}
        for k, v in infos.items():
            print(f"columns de type: {k}")
            display(df.loc[:, v].head())
            print("\n")
        print("Les valeurs manquantes à plus de 50%: ")
        manq= df.isna().sum()/df.shape[0]
        manq2= list(manq[manq>=0.5].index)
        display(manq2)
        print("\n")
        print("Unicité de modalité par variables: ")
        unique= df.nunique()
        uniq1= list(unique[unique==1].index)
        print("les variables avec des modalités uniques sont au nombre de "+str(len(uniq1)))
        print("\n")

    elif version==2:
        print("Analyse des doublons")
        dbl=["Code type local", "Date mutation", "Code postal", "Commune", "Valeur fonciere", "Surface terrain"]
        x=df[dbl]
        display(x.head())
        print(f"Il y a {x.shape[0]} lignes en qui se repetent")




#Bivariate analysis
def bivariate_analysis(df):
    print("Corrélation des variables quantitatives")
    corr= corr_map(df)
    quanti= df[corr.index]
    plt.title("Correlation des valeurs quantitatives")
    plt.show()
    print("Affichage des données quantitatives")
    display(quanti)
    print("\n")
    print("Comparaison Type local et Code Type local")
    display(df[["Type local", "Code type local"]].dropna().groupby(["Type local", "Code type local"], group_keys=False).apply(lambda x:x).head())
    print("\n")
    print("Comparaison Code commune et Commune")
    display(df[["Code commune", "Commune"]].dropna().groupby("Code commune").agg("count").head())
    print("\n")
    print("Comparaison code voie et Voie")
    display(df[["Code voie", "Voie"]].dropna().groupby("Code voie").agg("count").head())
    print("\n")






#traitement 
### version 1: sur les premières remarques, 
### version 2: sur les deuxiemes remarques
#Permet de préparer la base de données
def traitement(df, version=1):
    if(version==1): 
        #convertir valeurs foncieres
        if df["Valeur fonciere"].dtype != float:
            df["Valeur fonciere"]= df["Valeur fonciere"].str.replace(",", ".").astype("float")
        #suppression des columns à perc de valeurs null
        manq= handleNA(df, 0.6) #["Identifiant de document", "Reference document", "1 Articles CGI", "2 Articles CGI", "3 Articles CGI", "4 Articles CGI", "5 Articles CGI", "Identifiant local"]
        if len(manq)>0:
            if manq[0] in df.columns:
                    df= df.drop(columns=manq)
        #suppression des lignes dont la valeur foncière est nulle à 0 ou 1
        df= df[~df["Valeur fonciere"].isna()]
        df= df[~df["Valeur fonciere"].isin([0, 1])]
    elif version==2:
        #suppression des doublons sur la base des variables "Code type local", "Date mutation", "Code postal", "Commune", "Valeur fonciere", "Surface terrain"
        dups= [x for x in ["Code type local", "Date mutation", "Code postal", "Commune", "Valeur fonciere", "Surface terrain"] if x in df.columns]
        if "Code postal" in df.columns:
            df= df.drop_duplicates(subset=dups)
        
        #supprimer les variables qui se repètent
        repet= [x for x in ["Type local", "Code commune", "Code voie", "No plan", "Section", "Voie"] if x in df.columns ]
        if repet[0] in df.columns:
            df= df.drop(columns=repet)
    elif version==3:
        #drop les na de Voie, Code postal et section
        #df=df.dropna(subset=["Voie"])
        df=df.dropna(subset=["Code postal", "Nature culture", "Surface terrain", "Nombre pieces principales", "Surface reelle bati"])
        #df=df.dropna(subset=["Section"])
        #Imputation de surface terrain par median
        #df["Surface terrain"]=SimpleImputer(strategy="median").fit_transform(df[["Surface terrain"]])
        #Imputation Nombre pieces principales par le mode
        #df["Nombre pieces principales"]= SimpleImputer(strategy="most_frequent").fit_transform(df[["Nombre pieces principales"]])
        #imputation de Surface reelle bati par moyenne
        #df["Surface reelle bati"]= SimpleImputer(strategy="mean").fit_transform(df[["Surface reelle bati"]])
        #drop columns type de Voie, et No voie
        if "Type de voie" in df.columns:
            df= df.drop(columns=["Type de voie", "No voie"])
        df= df.dropna(subset=["Code type local"])
        #date de mutation scinder en mois jour annee
        df= data_f_date(df)
        #convertion du code en categorie
        df["Code departement"]= df["Code departement"].astype("str")
    return df


def encoding(df):
    lb= LabelEncoder()
    # "Commune", "Section" et section encoder par label
    df["Commune"]= lb.fit_transform(df["Commune"])
    #les autres par onehot
    df["Code departement"]= df["Code departement"].astype("str")
    o=df.select_dtypes("object")
    df= pd.get_dummies(df, columns=o.columns, dtype="int")#, dtype="int"
    return df

def classif_local(df):
    print("on regroupe les valeurs na du type local comme nos données à classer et les autres pour former le modèle")
    data1= df[~df["Code type local"].isna()]
    data2= df[df["Code type local"].isna()]
    print(f"les données à predire ont la taille {data2.shape[0]}x{data2.shape[1]}")
    print(f"les données à entrainer ont la taille {data1.shape[0]}x{data1.shape[1]}")
    print("\n")
    print("on scinde la base d'entrainement en X et Y")
    Y= data1["Code type local"]
    X= data1.drop(columns=["Code type local"])
    print("X: ")
    display(X.head())
    print("\n")
    print("Y: ")
    display(Y.head())
    print("\n")
    print("On centre réduit X et le split en train et test")
    xtrain

    return X


#la pipeline de nettoyage
def pipeline_nettoyage(df):
    df= traitement(df, 1)
    df= traitement(df, 2)
    df= traitement(df, 3)
    df= encoding(df)
    return df



#handle date 
##params:
### dataframe df
##return 
### dataframe (df2) with date extract as mois jour annee
### plot du total de la valeur fonciere par mois

def data_f_date(df):
    #get "Date mutation" and "Valeur fonciere"
    if ("Date mutation" in df.columns):
        if df["Date mutation"].dtype != "int":
            #convert date to string
            df["Date mutation"]= df["Date mutation"].astype("str")
            #split into "jour", "mois", "annee"
            df["jour"]= df["Date mutation"].agg(lambda x: x.split("/")[0]).astype("int")
            df["mois"]= df["Date mutation"].agg(lambda x: x.split("/")[1]).astype("int")
            df["annee"]= df["Date mutation"].agg(lambda x: x.split("/")[2]).astype("int")
            df= df.drop(columns=["Date mutation"])
    return df


#PLOT DE LA CORRELATION DES VALEURS NUMERIQUES SOUS FORME HEATMAP
##params df
##return corr
def corr_map(df):
    corr= df.corr(numeric_only=True)
    sn.heatmap(corr, cmap="Blues")
    return corr


def handleNA(df, perc):
    rp= (df.isna().sum()/df.shape[0]).sort_values(ascending= False)
    x= rp[rp>perc]
    return list(x.index)
