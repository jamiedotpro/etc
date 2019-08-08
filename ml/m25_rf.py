import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, KFold
import os

# 데이터 읽어 들이기
dir_path = os.getcwd()
winequality_white_file = os.path.join(dir_path, 'etc/data/winequality-white.csv')
wine = pd.read_csv(winequality_white_file, sep=';', encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

parameters = {
    'n_estimators': [100, 200],
    'max_depth': [2, 3, 4],
    'n_jobs': [-1]
}

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print('최적의 매개 변수 =', clf.best_estimator_)

y_pred = clf.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)

# 최적의 매개 변수 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=2, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
# 최종 정답률 =  0.9387755102040817
# 최종 정답률 =  0.9387755102040817