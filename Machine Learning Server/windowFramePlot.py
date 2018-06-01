import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import peakutils

COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis','activity']
p1 = pd.read_csv('1.csv', header=None, names=COLUMNS)
#RUNNING = pd.read_csv('Walking.csv', header=None, names=COLUMNS)[:3000]



def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)



def windows(df):
    start = 0
    chng=0
    i=0
    size=df.count()
    size-=1
    dir=1 #assuming postive slope
    #print('Inside windows function')
    while (start < size and i<size):
        
        if(df[i]>df[i+1] and dir>0):
            #print('change 1')
            chng+=1
            dir*=-1
        
        elif(df[i]<=df[i+1] and dir<0):
            #print('change 2')
            chng+=1
            dir*=-1
        if (chng==2):  
            #print('returning window size')
            #print(start)
            #print(i)
            yield start, start + i
            start= i
            chng=0
            #dir=1
        i+=1


# FINDING WINDOW SIZE TO EXTRACT FEATURE
#def windows(df, size=100):
 #   start = 0
  #  ct =0
   
  
   # if df[ct]<=df[ct+1]:direction=1
   # else: direction=-1
    #print(direction)
   # while start < df.count():
        
      #  flag=0
       # while ct<df.count():
        #    cur=ct+1
         #   nxt=ct+2
          #  if df[ct]<df[cur]:
           #     if df[cur]>df[nxt]:
            #        flag=+1
             #       if flag==1 : 
              #          ct=cur
               #         break
          #  if df[ct]>df[cur]:
           #     if df[cur]<df[nxt]:
            #        flag=+1
             #       if flag==2 : 
              #          ct=cur
               #         break
           # ct=ct+1        
       # print()
      #  indices = peakutils.indexes(df, thres=0.02/max(df), min_dist=0.1) 
       # print(indices)
        #while(flag<2):
            #while((df[ct]<=df[ct+1] and direction==1)or (df[ct]>df[ct+1] and direction==-1)):ct+=1
          #  while(df[ct]<=df[ct+1] && direction=-1):ct+=1
          #  direction*=-1
          #  flag+=1
       # while(df[window])

        #yield start, start + ct
        #start +=ct
#
fig, ax = plt.subplots(nrows=1, figsize=(15, 3))
fig, ay = plt.subplots(nrows=1, figsize=(15, 3))

fig, az = plt.subplots(nrows=1, figsize=(15, 3))

plot_axis(ax, p1['timestamp'], p1['xAxis'], 'person1')
plot_axis(ay, p1['timestamp'], p1['yAxis'], 'person1')
plot_axis(az, p1['timestamp'], p1['zAxis'], 'person1')

#for (st, end) in windows(p1['xAxis']):
    #print(st)
    #print(end)
   # ax.axvline(p1['timestamp'][st], color='r')


plt.show()

#for (start, end) in windows(p1['xAxis']):
 #   print('about to print windows')
  #  print(start)