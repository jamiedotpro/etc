import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
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

# RandomizedSearchCV에서 사용할 매개 변수 --- (*1)
parameters = {
    'svm__C': [1, 10, 100, 1000],
    'svm__kernel': ['linear', 'rbf', 'sigmoid'],
    'svm__gamma': [0.001, 0.0001],
}

parameters2 = {
    'svc__C': [1, 10, 100, 1000],
    'svc__kernel': ['linear', 'rbf', 'sigmoid'],
    'svc__gamma': [0.001, 0.0001],
}


# RandomizedSearchCV --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# make_pipeline과 pipeline의 차이는 디폴트로 지정된 키 이름을 사용할지, 키 이름을 직접 입력할지의 차이임
# pipe = make_pipeline(MinMaxScaler(), model) # == pipe = Pipeline([('minmaxscaler', MinMaxScaler()), ('kerasclassifier', model)])

# 방법1 Pipeline
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
clf = RandomizedSearchCV(pipe, parameters, cv=kfold_cv)
#clf = RandomizedSearchCV(SVC(), parameters, cv=kfold_cv)

###################################
'''
# 방법2 make_pipeline
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(), SVC())
clf = RandomizedSearchCV(pipe, parameters2, cv=kfold_cv)
'''

clf.fit(x_train, y_train)

print('최적의 매개 변수 =', clf.best_estimator_)

# 최적의 매개 변수로 평가하기 --- (*3)
y_pred = clf.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)


# 최적의 매개 변수 = Pipeline(memory=None,
#      steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svm', SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='linear',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False))])
# 최종 정답률 =  1.0
# 최종 정답률 =  1.0


