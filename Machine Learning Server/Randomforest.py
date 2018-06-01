
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.externals import joblib



dataset = np.loadtxt('ExtFeatures_1_2___001.csv', delimiter=",")


X = dataset[:, 1:]
y = dataset[:, 0]

c = RandomForestClassifier()

results = []
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    c.fit(X_train, y_train)
    res = c.score(X_test, y_test)
    print ('Loop', i, res)
    results.append(res)
joblib.dump(c, "random_model.pkl")

#print (np.mean(results), np.std(results))
