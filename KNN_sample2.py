#評価
test_data = pd.read_table("test.tsv")
test_data = DataFrame(test_data)
test_data_id = test_data




#標準化
scaler2 = StandardScaler()
scaler2.fit(test_data)
test_data = scaler2.transform(test_data)
test_data = DataFrame(test_data)

#主成分分析
pca2 = PCA(n_components=2)
test_data_pca2 = pca2.fit_transform(test_data)
test_prediction = knn_classfication.predict(test_data_pca2)
print(test_prediction)
#knn_classfication
#best_model

# x(元データ)で検証
#標準化
scaler3 = StandardScaler()
scaler3.fit(x)
x_test_data = scaler2.transform(x)
x_test_data = DataFrame(x_test_data)

#主成分分析
pca3 = PCA(n_components=2)
test_data_pca3 = pca3.fit_transform(x_test_data)
test_prediction2_knn = knn_classfication.predict(test_data_pca3)
test_prediction2_grid = best_model.predict(test_data_pca3)
#test_prediction2_knn_x = knn_classfication.predict(x)
test_prediction2_grid_x = best_model.predict(x)
print(test_prediction2)

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
print(test_prediction2_knn[test_prediction2_knn==1])
print("======================")
print(test_prediction2_grid[test_prediction2_grid==1])
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
print(y2.shape)
print(y2[10:19])
print("i:{}".format(i))

print("one:{}".format(one))
print("zero:{}".format(zero))

with open('sample_writer_row17.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(test_column)
    writer.writerow(test_prediction)