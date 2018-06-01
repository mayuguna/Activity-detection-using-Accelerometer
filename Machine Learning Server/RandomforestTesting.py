
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.externals import joblib



#feature = pd.read_csv('ExtFeatures_0_1_2___001.csv', header=None)

feature = np.loadtxt('Eating_Testin_2.csv', delimiter=",")

 #feature.reshape(-1, 1)
results = []
randomForest = joblib.load("./random_model.pkl")

#feature=joblib.load("./ExtFeatures_Testing__Drinking___001.csv")

prediction=randomForest.predict(feature)


print (prediction)
