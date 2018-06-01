
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.externals import joblib



feature = np.loadtxt('ExtFeatures_Testing__Writingfinal___001.csv', delimiter=",")[:,1:]
#dataset =pd.read_csv('ExtFeatures.csv')



randomForest = joblib.load("./randomforestmodel.plk")
    #feature=pd.read_csv("./feature.csv")
prediction=randomForest.predict(feature)

print prediction

#prediction=randomForest.predict(feature)
#print 'Random Forest', np.mean(results), np.std(results)