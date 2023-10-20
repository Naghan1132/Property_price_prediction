from preprocessing_deploiement import preprocessing
from classification_deploiement import classification
from regression_deploiement import regression

import pandas as pd


submissions = pd.read_csv('../submissions.csv')
# 378 0000 individus

submissions_cleaned = preprocessing(submissions)

submissions_classified = classification(submissions_cleaned)

predictions = regression(submissions_classified)

print(predictions.head())