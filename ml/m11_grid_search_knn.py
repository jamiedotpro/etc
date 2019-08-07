import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os

# 붓꽃 데이터 읽어 들이기
dir_path = os.getcwd()
iris_data_file = os.path.join(dir_path, 'etc/data/iris2.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# 그리드 서치에서 사용할 매개 변수 --- (*1)
parameters = [
    {'n_neighbors': [3, 5, 7, 10], 'weights':['uniform'], 'algorithm':['auto', 'ball_tree']},
    {'n_neighbors': [3, 5, 7, 10], 'weights':['distance'], 'algorithm':['auto', 'kd_tree']},
    {'n_neighbors': [3, 5, 7, 10], 'weights':['distance'], 'algorithm':['ball_tree', 'brute']},
]

# 그리드 서치 --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print('최적의 매개 변수 =', clf.best_estimator_)

# 최적의 매개 변수로 평가하기 --- (*3)
y_pred = clf.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)

# 최적의 매개 변수 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=None, n_neighbors=5, p=2,
#            weights='uniform')
# 최종 정답률 =  1.0
# 최종 정답률 =  1.0