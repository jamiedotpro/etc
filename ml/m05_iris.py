import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# 붓꽃 데이터 읽어 들이기
dir_path = os.getcwd()
iris_data_file = os.path.join(dir_path, 'etc/data/iris.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8', header=None,
                        names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name'])

print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
# loc는 레이블로 분리
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# iloc는 컬럼의 순서로 분리
# x2 = iris_data.iloc[:, 0:4]
# y2 = iris_data.iloc[:, 4]

print(x.shape)
print(y.shape)

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# print(y_test)   # str형식으로 저장되어 있다

# print(x_train.shape)
# print(x_test.shape)

# 학습하기
# clf = SVC()   # 분류모델
# clf = KNeighborsClassifier()    # 분류모델
clf = LinearSVC()
clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_pred))
