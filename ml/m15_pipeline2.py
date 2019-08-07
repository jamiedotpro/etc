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
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.0001],
}


# RandomizedSearchCV --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#!! pipeline 수정필요
# 방법1
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])

clf = RandomizedSearchCV(pipe, parameters, cv=5)
#clf = RandomizedSearchCV(SVC(), parameters, cv=kfold_cv)

clf.fit(x_train, y_train)


# # 방법1-1
# clf = RandomizedSearchCV(SVC(), parameters, cv=kfold_cv)
# pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
# pipe.fit(x_train, y_train)
# print(pipe.score(x_train, y_train))
# import sys
# sys.exit()

###################################

# 방법2
# from sklearn.pipeline import make_pipeline
# clf = RandomizedSearchCV(SVC(), parameters2, cv=kfold_cv)
# pipe = make_pipeline(MinMaxScaler(), clf)
# pipe.fit(x_train, y_train)  # clf.fit이 내부에서 처리됨
# pipe.score(x_train, y_train)

###################################
# pipe = Pipeline([('vec', CountVectorizer()), ('clf', LogisticRegression()])
# param_grid = [{'clf__C': [1, 10, 100, 1000]}
# gs = GridSearchCV(pipe, param_grid)
# gs.fit(X, y)
###################################
# pipe = make_pipeline(CountVectorizer(), LogisticRegression())     
# param_grid = [{'logisticregression__C': [1, 10, 100, 1000]}
# gs = GridSearchCV(pipe, param_grid)
# gs.fit(X, y)
###################################

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


