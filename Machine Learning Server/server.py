from flask import Flask, request, jsonify
from sklearn.externals import joblib
from flask import jsonify 
import matplotlib
from statistics import mode
import math
import numpy as np
import pandas as pd
import sys
import csv
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
import numpy as np
import time

app = Flask(__name__)

# root
@app.route("/predict")
def index():
    time.sleep(15)
    #feature extraction    
    COLUMNS = ['id','time', 'x_axis', 'y_axis', 'z_axis']
    DATASET = pd.read_csv('http://172.20.10.2:8080/DailyTracker/predict.csv', names=COLUMNS,skipfooter=1)
    #DATASET = pd.read_csv('./predict.csv', names=COLUMNS)
    print DATASET.head()
    ## maginitude funtion to normalise the data
    def magnitude(activity):
        x2 = activity['x_axis'] * activity['x_axis']
        y2 = activity['y_axis'] * activity['y_axis']
        z2 = activity['z_axis'] * activity['z_axis']
        m2 = x2 + y2 + z2
        m = m2.apply(lambda x: math.sqrt(x))
        return m
    #adding magnitude to the variable
    DATASET['magnitude'] = magnitude(DATASET)
    def windows(df, size=50):
        start = 0
        while start < df.count():
            yield start, start + size
            start += (size / 2)
    def jitter(axis, start, end):
        j = float(0)
        for i in xrange(start, min(end, axis.count())):
            if start != 0:
                j += abs(axis[i] - axis[i-1])
        return j / (end-start)

    def mean_crossing_rate(axis, start, end):
        cr = 0
        m = axis.mean()
        for i in xrange(start, min(end, axis.count())):
            if start != 0:
                p = axis[i-1] > m
                c = axis[i] > m
                if p != c:
                    cr += 1
        return float(cr) / (end-start-1)

    def window_summary(axis, start, end):
        acf = stattools.acf(axis[start:end])
        acv = stattools.acovf(axis[start:end])
        sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
        return [
            jitter(axis, start, end), #singal processing
            mean_crossing_rate(axis, start, end), # rate with signal crossing the mean value
            axis[start:end].mean(),
            axis[start:end].std(),
            axis[start:end].var(),
            axis[start:end].min(),
            axis[start:end].max(),
            acf.mean(), # mean auto correlation
            acf.std(), # standard deviation auto correlation
            acv.mean(), # mean auto covariance
            acv.std(), # standard deviation auto covariance
            skew(axis[start:end]),
            kurtosis(axis[start:end]),
            math.sqrt(sqd_error.mean())
        ]

    def features(activity):
        for (start, end) in windows(activity['time']):
            features = []
            for axis in ['x_axis', 'y_axis', 'z_axis', 'magnitude']:
                features += window_summary(activity[axis], start, end)
            yield features

    activities = [DATASET]

    with open('ExtFeatures_Testing__Writingfinal___001.csv', 'w') as out:
        rows = csv.writer(out)
        for i in range(0, len(activities)):
            for f in features(activities[i]):
                rows.writerow([i]+f)


    
	randomForest = joblib.load("./randomforestmodelfinal_4.plk")
    #feature=pd.read_csv("./feature.csv")
    #prediction=randomForest.predict(feature)
    #conn = mysql.connect()
    #cursor =conn.cursor()
    #cursor.execute("select * from healthbot.users where  username = 'ab123'")
    #data = cursor.fetchone()
    #print data
    #result=prediction.mode()  
   
    feature = np.loadtxt('ExtFeatures_Testing__Writingfinal___001.csv', delimiter=",")[:,1:]
    #dataset =pd.read_csv('ExtFeatures.csv')
    #feature=dataset[:,1:]
    #randomForest = joblib.load("./randomforestmodel.plk")
    #feature=pd.read_csv("./feature.csv")
    #feature.iloc[::2]
    prediction=randomForest.predict(feature)
    #prediction=randomForest.predict(feature)
    #print 'Random Forest', np.mean(results), np.std(results)
    print prediction
    result=mode(prediction)
    if result==1:
        return "Drinking"
    if result==2:
        return "Writing"
    if result==3:
        return "Walking"    
    return "UnKnown Activity" 		   
		

# POST
@app.route("/test")
def get_text_prediction():
    return "Drinking"

