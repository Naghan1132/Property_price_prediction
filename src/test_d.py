import pandas as pd

df = pd.read_csv('../submissions.csv')
from preprocessing_deploiement import *

df.preprocessing(df)