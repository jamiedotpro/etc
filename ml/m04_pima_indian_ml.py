import os
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

# 1. 데이터
dir_path = os.getcwd()
pima_indians_file = os.path.join(dir_path, 'etc/data/pima-indians-diabetes.csv')
dataset = np.loadtxt(pima_indians_file, delimiter=',')
x_data = dataset[:, 0:8]
y_data = dataset[:, 8]

# 2. 모델
# model = LinearSVC() # acc =  0.6809895833333334
# model = SVC() # acc =  1.0
# model = KNeighborsClassifier(n_neighbors=1) # acc =  1.0
model = KNeighborsRegressor(n_neighbors=1)  # acc =  1.0

# 3. 실행
model.fit(x_data, y_data)

# 4. 평가 예측
x_test = x_data
y_test = y_data
y_predict = model.predict(x_test)

print(x_test, '의 예측결과: \n', y_predict)
print('acc = ', accuracy_score(y_test, y_predict))

# import sys
# sys.exit()
