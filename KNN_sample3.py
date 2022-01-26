import pandas as pd
from pandas import DataFrame
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm,metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import joblib
!pip install mlxtend
from mlxtend.frequent_patterns import apriori
#from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
!pip install -U imbalanced-learn


train_data = pd.read_table("train.tsv")
#x = pd.read_table("test.tsv")

train_data=DataFrame(train_data)

#print("テストサイズ：",test_data.shape)
print(train_data.columns.values)
print(train_data.shape)
print(train_data.tail())

from sklearn.model_selection import train_test_split
x = DataFrame(train_data.drop("bot",axis=1))
y = DataFrame(train_data["bot"])


#前処理
print("前処理")
print("=================")
print("欠損値")
print(x.isnull().sum())
print("=================")
print("データ：平均")
print(round(x.mean()))
print("=================")
print("データ：分散")
print(round(x.std()))
print("================")

#データ(標準化用:x2,外れ値用:x3,両方:x4)
x2 = x
x3 = x
#:x4

#標準化:x2
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x2)
x2 = scaler.transform(x2)
x2 = DataFrame(x2)
print("==================")
print("x2:平均は")
print(round(x2.mean()))
print("x2:分散は")
print(round(x2.std()))
print("=========================================")
print("x2:列名")
print(x2.columns)
print("x2[1]===============================================")
print(x2[1].head)
x5 = DataFrame(x2.loc[:,[1, 2, 3, 4, 5, 6, 7, 8, 11, 13]])
print(x5.shape)
print("x2[1]===========================================")
print(x5.head)
#x5=x2[1, 2, 3, 4, 5, 6, 7, 8, 11, 13]







#train用とtest用に分ける
X_train, X_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, random_state=0)


#KNN
knn_classfication = KNeighborsClassifier()
knn_classfication.fit(X_resampled,y_resampled)
knn_prediction = knn_classfication.predict(X_test)
knn_collect_score = metrics.accuracy_score(y_test, knn_prediction)
print("=========================================")
print("KNNの正解率 =", knn_collect_score)
print("=========================================")

#評価
test_data = pd.read_table("test.tsv")
test_data = DataFrame(test_data)
test_data_id = test_data

#前処理
print("前処理")
print("=================")
print("欠損値")
print(test_data.isnull().sum())
print("=================")
print("データ：平均")
print(round(test_data.mean()))
print("=================")
print("データ：分散")
print(round(test_data.std()))
print("================")

#標準化
scaler2 = StandardScaler()
scaler2.fit(test_data)
test_data2 = scaler2.transform(test_data)
test_data2 = DataFrame(test_data2)

#予測(評価用データ)
#標準化あり
test_prediction_knn2 = knn_classfication.predict(test_data2)
test_prediction_grid2 = best_model.predict(test_data2)
#標準化なし
test_prediction_knn1 = knn_classfication.predict(test_data)
test_prediction_grid1 = best_model.predict(test_data)

#knn_classfication
#best_model


# x(元データ)で検証
#標準化
scaler3 = StandardScaler()
scaler3.fit(x)
x_test_data = scaler2.transform(x)
x_test_data = DataFrame(x_test_data)

#予測
test_prediction2_knn = knn_classfication.predict(x_test_data)
test_prediction2_grid = best_model.predict(x_test_data)
test_prediction2_knn_x = knn_classfication.predict(x)
test_prediction2_grid_x = best_model.predict(x)
#print(test_prediction2)

import csv
import pprint

test_column = test_data_id["id"]

print(test_column)
print("======================")
#sugukesu
train_data3 = pd.read_table("train.tsv")
y2 = DataFrame(train_data3["bot"])
zero=0
one=0
print("test_prediction_knn1")
print(test_prediction_knn1[test_prediction_knn1==1])
print("======================")
print("test_prediction_grid1")
print(test_prediction_grid1[test_prediction_grid1==1])

print("======================")
"""
for i in y2:
  if i == 0:
    zero=zero+1
  if i == 1:
    one=one+1
  print(i)

"""

print("======================")

with open('test_prediction_knn1.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(test_column)
    writer.writerow(test_prediction_knn1)
