# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:57:25 2019

@author: Islem
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:59:41 2019

@author: Mr Dever
"""
##### Impoting libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
###### Preprocessing
df=pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?',-99999, inplace=True) #-99999 to make the "?" an outlier
df.drop(['id'], 1, inplace=True)# to make the id an outlier because it's useless for the class

###### Training
X=np.array(df.drop(['class'],1))
Y=np.array(df['class'])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

clf=KNeighborsClassifier(n_neighbors=3) #GaussianNB()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print("Accuracy: ",accuracy)

a=[[4,2,1,1,1,2,3,2,1],[8,8,7,4,10,10,7,8,7]]
example_measures=np.array(a)
example_measures=example_measures.reshape(len(example_measures),-1)#len here is 2, using len to make it more dynamic and auto changeble 

prediction=clf.predict(example_measures)
print("Class: ",prediction, "|2 for benign, 4 for malignant")
