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

#x3
#外れ値処理(関数)：四分位範囲より◯倍（default=1.5）離れたデータを平均値に変更
def outlier_iqr(df,multiple_number=1.5, columns=None):
  if columns == None:
    columns = df.columns
  
  for col in columns:
    q1 = df[col].describe()['25%']
    q3 = df[col].describe()['75%']
    iqr = q3 - q1
    outline_min = q1 - iqr * multiple_number
    outline_max = q3 + iqr * multiple_number

    mean2 = df[col].describe()['50%']    
    #変換
    for i in df[(df[col] < outline_min)].index.values:
      df[col][i] = mean2
    
    #変換
    for i in df[(df[col] > outline_max)].index.values:
      df[col][i] = mean2
  return df

#外れ値処理：関数実行(x3)
x3 = outlier_iqr(x3,multiple_number=1.0)
print("xのデータ数測定")
print(x.shape)
print("x3のデータ数測定")
print(x3.shape)
print("x3の格特徴量ごとのデータ数確認")
for colum in x3.columns:
  print(x3[colum].shape)
print("欠損値")
print(x3.isnull().sum())

#外れ値(箱ヒゲ図で確認):x3
for i in x3.columns:
  plt.figure(figsize=(10,10))
  sns.boxplot(x3[i],orient='v',width=0.5)

#外れ値、標準化:x4
x4 = x2
x4 = outlier_iqr(x4,multiple_number=1.0)

#主成分分析
from sklearn.decomposition import PCA
pca = PCA(n_components=15)
#x2_pca = pca.fit_transform(x2,y)
#x3_pca = pca.fit_transform(x3,y)
x4_pca = pca.fit_transform(x4,y)
#x4_pca = pd.DataFrame(x4_pca)
#x3_pca = pd.DataFrame(x3_pca)
print("=========================================")
print("xの特徴量",x.columns)
print("x2の特徴量",x2.columns)
print("x3の特徴量",x3.columns)
print("x4の特徴量",x4.columns)
#print("主成分分析の特徴量",x3_pca.columns)
print("=========================================")


#train用とtest用に分ける
X_train, X_test, y_train, y_test = train_test_split(x4_pca, y, test_size=0.2, random_state=0)

#KNN
knn_classfication = KNeighborsClassifier()
knn_classfication.fit(X_train,y_train)
knn_prediction = knn_classfication.predict(X_test)
knn_collect_score = metrics.accuracy_score(y_test, knn_prediction)
print("=========================================")
print("KNNの正解率 =", knn_collect_score)
print("=========================================")

#Grid_search
from sklearn.model_selection import GridSearchCV
estimator = KNeighborsClassifier()
param_grid = [{
    'n_neighbors' : [50,80,100,200,500],
    'weights' : ['uniform','distance'],
    'algorithm':['ball_tree','kd_tree','brute'],
    'leaf_size' : [1,10,30,50,70,100],
    'p' : [1,2],
    'metric' : ['euclidean','manhattan','chebyshev','minkowski','wminkowski','seuclidean','mahalanobis']
}]
cv=5
tuned_model = GridSearchCV(estimator=estimator,
                           param_grid=param_grid,
                           cv=cv,
                           return_train_score=False
                           )

tuned_model.fit(X_train,y_train)

#結果(外れ値は四分位範囲から1.5倍以上離れてるものを中央値に置き換え)
#(1)標準化、外れ値処理なし => KNNの正解率 = 0.8207547169811321
#(2)標準化あり、外れ値処理なし => KNNの正解率 = 0.8238993710691824
#(3)標準化なし、外れ値処理あり => KNNの正解率 = 0.8207547169811321 
#(4)標準化、外れ値処理あり　=> KNNの正解率 = 0.8238993710691824

#主成分分析(5)
#(5) (2)の主成分分析主成分分析ver. => KNNの正解率 = 0.8364779874213837
#(5) (3)の主成分分析主成分分析ver. => KNNの正解率 = 0.8238993710691824
#(5) (4)の主成分分析主成分分析ver. => KNNの正解率 =  0.8364779874213837

#grid_search後(6) 2次元
#(6) (5)(2) KNN = 0.8459119496855346 
#(6) (5)(3) KNN = 0.8459119496855346
#(6) (5)(4) KNN = 0.8459119496855346


#grid_search 2次元次元 KNN = 0.8584905660377359
#grid_search 3次元次元 KNN = 0.8553459119496856
#grid_search 4次元次元 KNN = 0.8522012578616353
#grid_search 5次元次元 KNN = 0.8584905660377359
#grid_search 6次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8427672955974843
"""
#grid_search 7次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8522012578616353
"""
#grid_search 8次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8522012578616353
"""
#grid_search 9次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8522012578616353
"""
#grid_search 10次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8522012578616353
"""
#grid_search 11次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8553459119496856
"""
#grid_search 12次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8553459119496856
"""
#grid_search 13次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8553459119496856
"""
#grid_search 14次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8553459119496856
"""
#grid_search 15次元次元 
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 80, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8553459119496856
"""
"""
{'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 50, 'p': 1, 'weights': 'distance'}
Grid_Search(train) =  1.0
Grid_Search(test) =  0.8459119496855346
"""

#12/21
#主成分分析の次元数を増やして精度を測定したが、6以降、過学習していることから、学習用データの割合を増やした（test_sizeをを0.1にした）が収まらなかったため、
#次回は正則化を用いて過学習を抑える


#まとめ
#(1)外れ値の置き換えをデータの特徴から考え工夫する
#(2)default_profile_imageのデータについて考慮する
#(3)教師データを標準化していなかった＝＞教師データはやらなくて良い

  pd.DataFrame(tuned_model.cv_results_).T
  
  print(tuned_model.best_params_)
best_model = tuned_model.best_estimator_
print("Grid_Search(train) = ",best_model.score(X_train,y_train))
print("Grid_Search(test) = ",best_model.score(X_test,y_test))