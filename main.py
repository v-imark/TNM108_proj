import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statistics
from math import pi
from math import e
from sklearn import preprocessing


data = pd.read_csv('MonkeyPoxData.csv')
target = data.MonkeyPox
print(data.columns.values)

train = data.iloc[:int(len(data.index)/2), :]
test = data.iloc[int(len(data.index)/2):, :]

train = train.drop(['Patient_ID'], axis=1)
test = test.drop(['Patient_ID'], axis=1)

labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(train['MonkeyPox'])
labelEncoder.fit(test['MonkeyPox'])
train['MonkeyPox'] = labelEncoder.transform(train['MonkeyPox'])
test['MonkeyPox'] = labelEncoder.transform(test['MonkeyPox'])

print(train[["Systemic Illness", "MonkeyPox"]].groupby(['Systemic Illness'], as_index=False).mean().sort_values(by='MonkeyPox',ascending=False))

#X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.3)






